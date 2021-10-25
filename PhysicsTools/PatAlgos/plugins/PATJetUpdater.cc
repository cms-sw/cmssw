/**
  \class    pat::PATJetUpdater PATJetUpdater.h "PhysicsTools/PatAlgos/interface/PATJetUpdater.h"
  \brief    Produces pat::Jet's

   The PATJetUpdater produces analysis-level pat::Jet's starting from
   a collection of pat::Jet's and updates information.

  \author   Andreas Hinzmann
  \version  $Id: PATJetUpdater.h,v 1.00 2014/03/11 18:13:54 srappocc Exp $
*/

#include "CommonTools/Utils/interface/PtComparator.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/JetCorrFactors.h"
#include "DataFormats/PatCandidates/interface/UserData.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/transform.h"
#include "PhysicsTools/PatAlgos/interface/PATUserDataHelper.h"

#include <algorithm>
#include <memory>
#include <vector>

namespace pat {

  class PATJetUpdater : public edm::stream::EDProducer<> {
  public:
    explicit PATJetUpdater(const edm::ParameterSet& iConfig);
    ~PATJetUpdater() override;

    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    // configurables
    edm::EDGetTokenT<edm::View<reco::Jet>> jetsToken_;
    bool sort_;
    bool addJetCorrFactors_;
    std::vector<edm::EDGetTokenT<edm::ValueMap<JetCorrFactors>>> jetCorrFactorsTokens_;

    bool addBTagInfo_;
    bool addDiscriminators_;
    std::vector<edm::InputTag> discriminatorTags_;
    std::vector<edm::EDGetTokenT<reco::JetFloatAssociation::Container>> discriminatorTokens_;
    std::vector<std::string> discriminatorLabels_;
    bool addTagInfos_;
    std::vector<edm::InputTag> tagInfoTags_;
    std::vector<edm::EDGetTokenT<edm::View<reco::BaseTagInfo>>> tagInfoTokens_;
    std::vector<std::string> tagInfoLabels_;

    GreaterByPt<Jet> pTComparator_;

    bool useUserData_;
    pat::PATUserDataHelper<pat::Jet> userDataHelper_;
    //
    bool printWarning_;  // this is introduced to issue warnings only once per job
  };

}  // namespace pat

using namespace pat;

PATJetUpdater::PATJetUpdater(const edm::ParameterSet& iConfig)
    : useUserData_(iConfig.exists("userData")), printWarning_(iConfig.getParameter<bool>("printWarning")) {
  // initialize configurables
  jetsToken_ = consumes<edm::View<reco::Jet>>(iConfig.getParameter<edm::InputTag>("jetSource"));
  sort_ = iConfig.getParameter<bool>("sort");
  addJetCorrFactors_ = iConfig.getParameter<bool>("addJetCorrFactors");
  if (addJetCorrFactors_) {
    jetCorrFactorsTokens_ = edm::vector_transform(
        iConfig.getParameter<std::vector<edm::InputTag>>("jetCorrFactorsSource"),
        [this](edm::InputTag const& tag) { return mayConsume<edm::ValueMap<JetCorrFactors>>(tag); });
  }
  addBTagInfo_ = iConfig.getParameter<bool>("addBTagInfo");
  addDiscriminators_ = iConfig.getParameter<bool>("addDiscriminators");
  discriminatorTags_ = iConfig.getParameter<std::vector<edm::InputTag>>("discriminatorSources");
  discriminatorTokens_ = edm::vector_transform(discriminatorTags_, [this](edm::InputTag const& tag) {
    return mayConsume<reco::JetFloatAssociation::Container>(tag);
  });
  addTagInfos_ = iConfig.getParameter<bool>("addTagInfos");
  tagInfoTags_ = iConfig.getParameter<std::vector<edm::InputTag>>("tagInfoSources");
  tagInfoTokens_ = edm::vector_transform(
      tagInfoTags_, [this](edm::InputTag const& tag) { return mayConsume<edm::View<reco::BaseTagInfo>>(tag); });
  if (discriminatorTags_.empty()) {
    addDiscriminators_ = false;
  } else {
    for (std::vector<edm::InputTag>::const_iterator it = discriminatorTags_.begin(), ed = discriminatorTags_.end();
         it != ed;
         ++it) {
      std::string label = it->label();
      std::string::size_type pos = label.find("JetTags");
      if ((pos != std::string::npos) && (pos != label.length() - 7)) {
        label.erase(pos + 7);  // trim a tail after "JetTags"
      }
      if (!it->instance().empty()) {
        label = (label + std::string(":") + it->instance());
      }
      discriminatorLabels_.push_back(label);
    }
  }
  if (tagInfoTags_.empty()) {
    addTagInfos_ = false;
  } else {
    for (std::vector<edm::InputTag>::const_iterator it = tagInfoTags_.begin(), ed = tagInfoTags_.end(); it != ed;
         ++it) {
      std::string label = it->label();
      std::string::size_type pos = label.find("TagInfos");
      if ((pos != std::string::npos) && (pos != label.length() - 8)) {
        label.erase(pos + 8);  // trim a tail after "TagInfos"
      }
      tagInfoLabels_.push_back(label);
    }
  }
  if (!addBTagInfo_) {
    addDiscriminators_ = false;
    addTagInfos_ = false;
  }
  // Check to see if the user wants to add user data
  if (useUserData_) {
    userDataHelper_ = PATUserDataHelper<Jet>(iConfig.getParameter<edm::ParameterSet>("userData"), consumesCollector());
  }
  // produces vector of jets
  produces<std::vector<Jet>>();
  produces<edm::OwnVector<reco::BaseTagInfo>>("tagInfos");
}

PATJetUpdater::~PATJetUpdater() {}

void PATJetUpdater::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Get the vector of jets
  edm::Handle<edm::View<reco::Jet>> jets;
  iEvent.getByToken(jetsToken_, jets);

  // read in the jet correction factors ValueMap
  std::vector<edm::ValueMap<JetCorrFactors>> jetCorrs;
  if (addJetCorrFactors_) {
    for (size_t i = 0; i < jetCorrFactorsTokens_.size(); ++i) {
      edm::Handle<edm::ValueMap<JetCorrFactors>> jetCorr;
      iEvent.getByToken(jetCorrFactorsTokens_[i], jetCorr);
      jetCorrs.push_back(*jetCorr);
    }
  }

  // Get the vector of jet tags with b-tagging info
  std::vector<edm::Handle<reco::JetFloatAssociation::Container>> jetDiscriminators;
  if (addBTagInfo_ && addDiscriminators_) {
    jetDiscriminators.resize(discriminatorTokens_.size());
    for (size_t i = 0; i < discriminatorTokens_.size(); ++i) {
      iEvent.getByToken(discriminatorTokens_[i], jetDiscriminators[i]);
    }
  }
  std::vector<edm::Handle<edm::View<reco::BaseTagInfo>>> jetTagInfos;
  if (addBTagInfo_ && addTagInfos_) {
    jetTagInfos.resize(tagInfoTokens_.size());
    for (size_t i = 0; i < tagInfoTokens_.size(); ++i) {
      iEvent.getByToken(tagInfoTokens_[i], jetTagInfos[i]);
    }
  }

  // loop over jets
  auto patJets = std::make_unique<std::vector<Jet>>();

  auto tagInfosOut = std::make_unique<edm::OwnVector<reco::BaseTagInfo>>();

  edm::RefProd<edm::OwnVector<reco::BaseTagInfo>> h_tagInfosOut =
      iEvent.getRefBeforePut<edm::OwnVector<reco::BaseTagInfo>>("tagInfos");

  for (edm::View<reco::Jet>::const_iterator itJet = jets->begin(); itJet != jets->end(); itJet++) {
    // construct the Jet from the ref -> save ref to original object
    unsigned int idx = itJet - jets->begin();
    const edm::RefToBase<reco::Jet> jetRef = jets->refAt(idx);
    const edm::RefToBase<Jet> patJetRef(jetRef.castTo<JetRef>());
    Jet ajet(patJetRef);

    if (addJetCorrFactors_) {
      // undo previous jet energy corrections
      ajet.setP4(ajet.correctedP4(0));
      // clear previous JetCorrFactors
      ajet.jec_.clear();
      // add additional JetCorrs to the jet
      for (unsigned int i = 0; i < jetCorrFactorsTokens_.size(); ++i) {
        const JetCorrFactors& jcf = jetCorrs[i][jetRef];
        // uncomment for debugging
        // jcf.print();
        ajet.addJECFactors(jcf);
      }
      std::vector<std::string> levels = jetCorrs[0][jetRef].correctionLabels();
      if (std::find(levels.begin(), levels.end(), "L2L3Residual") != levels.end()) {
        ajet.initializeJEC(jetCorrs[0][jetRef].jecLevel("L2L3Residual"));
      } else if (std::find(levels.begin(), levels.end(), "L3Absolute") != levels.end()) {
        ajet.initializeJEC(jetCorrs[0][jetRef].jecLevel("L3Absolute"));
      } else {
        ajet.initializeJEC(jetCorrs[0][jetRef].jecLevel("Uncorrected"));
        if (printWarning_) {
          edm::LogWarning("L3Absolute not found")
              << "L2L3Residual and L3Absolute are not part of the jetCorrFactors\n"
              << "of module " << jetCorrs[0][jetRef].jecSet() << ". Jets will remain"
              << " uncorrected.";
          printWarning_ = false;
        }
      }
    }

    // add b-tag info if available & required
    if (addBTagInfo_) {
      if (addDiscriminators_) {
        for (size_t k = 0; k < jetDiscriminators.size(); ++k) {
          float value = (*jetDiscriminators[k])[jetRef];
          ajet.addBDiscriminatorPair(std::make_pair(discriminatorLabels_[k], value));
        }
      }
      if (addTagInfos_) {
        for (size_t k = 0; k < jetTagInfos.size(); ++k) {
          const edm::View<reco::BaseTagInfo>& taginfos = *jetTagInfos[k];
          // This is not associative, so we have to search the jet
          edm::Ptr<reco::BaseTagInfo> match;
          // Try first by 'same index'
          if ((idx < taginfos.size()) && (taginfos[idx].jet() == jetRef)) {
            match = taginfos.ptrAt(idx);
          } else {
            // otherwise fail back to a simple search
            for (edm::View<reco::BaseTagInfo>::const_iterator itTI = taginfos.begin(), edTI = taginfos.end();
                 itTI != edTI;
                 ++itTI) {
              if (itTI->jet() == jetRef) {
                match = taginfos.ptrAt(itTI - taginfos.begin());
                break;
              }
            }
          }
          if (match.isNonnull()) {
            tagInfosOut->push_back(match->clone());
            // set the "forward" ptr to the thinned collection
            edm::Ptr<reco::BaseTagInfo> tagInfoForwardPtr(
                h_tagInfosOut.id(), &tagInfosOut->back(), tagInfosOut->size() - 1);
            // set the "backward" ptr to the original collection for association
            const edm::Ptr<reco::BaseTagInfo>& tagInfoBackPtr(match);
            // make FwdPtr
            TagInfoFwdPtrCollection::value_type tagInfoFwdPtr(tagInfoForwardPtr, tagInfoBackPtr);
            ajet.addTagInfo(tagInfoLabels_[k], tagInfoFwdPtr);
          }
        }
      }
    }

    if (useUserData_) {
      userDataHelper_.add(ajet, iEvent, iSetup);
    }

    // reassign the original object reference to preserve reference to the original jet the input PAT jet was derived from
    // (this needs to be done at the end since cloning the input PAT jet would interfere with adding UserData)
    ajet.refToOrig_ = patJetRef->originalObjectRef();

    patJets->push_back(ajet);
  }

  // sort jets in pt
  if (sort_) {
    std::sort(patJets->begin(), patJets->end(), pTComparator_);
  }

  // put genEvt  in Event
  iEvent.put(std::move(patJets));

  iEvent.put(std::move(tagInfosOut), "tagInfos");
}

// ParameterSet description for module
void PATJetUpdater::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription iDesc;
  iDesc.setComment("PAT jet producer module");

  // input source
  iDesc.add<edm::InputTag>("jetSource", edm::InputTag("no default"))->setComment("input collection");

  // sort inputs (by pt)
  iDesc.add<bool>("sort", true);

  // tag info
  iDesc.add<bool>("addTagInfos", true);
  std::vector<edm::InputTag> emptyVInputTags;
  iDesc.add<std::vector<edm::InputTag>>("tagInfoSources", emptyVInputTags);

  // jet energy corrections
  iDesc.add<bool>("addJetCorrFactors", true);
  iDesc.add<std::vector<edm::InputTag>>("jetCorrFactorsSource", emptyVInputTags);

  // btag discriminator tags
  iDesc.add<bool>("addBTagInfo", true);
  iDesc.add<bool>("addDiscriminators", true);
  iDesc.add<std::vector<edm::InputTag>>("discriminatorSources", emptyVInputTags);

  // silent warning if false
  iDesc.add<bool>("printWarning", true);

  // Check to see if the user wants to add user data
  edm::ParameterSetDescription userDataPSet;
  PATUserDataHelper<Jet>::fillDescription(userDataPSet);
  iDesc.addOptional("userData", userDataPSet);

  descriptions.add("PATJetUpdater", iDesc);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATJetUpdater);
