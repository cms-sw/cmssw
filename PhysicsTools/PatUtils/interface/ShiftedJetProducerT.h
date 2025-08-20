#ifndef PhysicsTools_PatUtils_ShiftedJetProducerT_h
#define PhysicsTools_PatUtils_ShiftedJetProducerT_h

/** \class ShiftedJetProducerT
 *
 * Vary energy of jets by +/- 1 standard deviation,
 * in order to estimate resulting uncertainty on MET
 *
 * NOTE: energy scale uncertainties are taken from the Database
 *
 * \author Christian Veelken, LLR
 *
 *
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "JetMETCorrections/JetCorrector/interface/JetCorrector.h"
#include "JetMETCorrections/Objects/interface/JetCorrectionsRecord.h"
#include "JetMETCorrections/Type1MET/interface/JetCorrExtractorT.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectionUncertainty.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "PhysicsTools/PatUtils/interface/PATJetCorrExtractor.h"
#include "PhysicsTools/PatUtils/interface/RawJetExtractorT.h"

#include <string>
#include <optional>

template <typename T, typename Textractor>
class ShiftedJetProducerT : public edm::stream::EDProducer<> {
  using JetCollection = std::vector<T>;

public:
  explicit ShiftedJetProducerT(const edm::ParameterSet& cfg)
      : moduleLabel_(cfg.getParameter<std::string>("@module_label")),
        src_(cfg.getParameter<edm::InputTag>("src")),
        srcToken_(consumes<JetCollection>(src_)),
        jetCorrPayloadName_(""),
        jecUncertainty_(),
        jecUncertaintyValue_() {
    if (cfg.exists("jecUncertaintyValue")) {
      jecUncertaintyValue_.emplace(cfg.getParameter<double>("jecUncertaintyValue"));
    } else {
      jetCorrUncertaintyTag_ = cfg.getParameter<std::string>("jetCorrUncertaintyTag");
      if (cfg.exists("jetCorrInputFileName")) {
        jetCorrInputFileName_ = cfg.getParameter<edm::FileInPath>("jetCorrInputFileName");
        if (jetCorrInputFileName_.location() == edm::FileInPath::Unknown)
          throw cms::Exception("ShiftedJetProducerT")
              << " Failed to find JEC parameter file = " << jetCorrInputFileName_ << " !!\n";
        JetCorrectorParameters jetCorrParameters(jetCorrInputFileName_.fullPath(), jetCorrUncertaintyTag_);
        jecUncertainty_.emplace(jetCorrParameters);
      } else {
        jetCorrPayloadName_ = cfg.getParameter<std::string>("jetCorrPayloadName");
        jetCorrPayloadToken_ = esConsumes(edm::ESInputTag("", jetCorrPayloadName_));
      }
    }

    addResidualJES_ = cfg.getParameter<bool>("addResidualJES");
    if (addResidualJES_) {
      if (cfg.exists("jetCorrLabelUpToL3")) {
        jetCorrLabelUpToL3_ = cfg.getParameter<edm::InputTag>("jetCorrLabelUpToL3");
        jetCorrTokenUpToL3_ = consumes<reco::JetCorrector>(jetCorrLabelUpToL3_);
      }
      if (cfg.exists("jetCorrLabelUpToL3Res")) {
        jetCorrLabelUpToL3Res_ = cfg.getParameter<edm::InputTag>("jetCorrLabelUpToL3Res");
        jetCorrTokenUpToL3Res_ = consumes<reco::JetCorrector>(jetCorrLabelUpToL3Res_);
      }
    }
    jetCorrEtaMax_ = cfg.getParameter<double>("jetCorrEtaMax");

    shiftBy_ = cfg.getParameter<double>("shiftBy");

    verbosity_ = cfg.getUntrackedParameter<int>("verbosity");

    putToken_ = produces<JetCollection>();

    //PATJetCorrExtractor wants a string not a product when called
    static_assert(not std::is_base_of<class PATJetCorrExtractor, Textractor>::value);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& iDesc) {
    //NOTE: the full complexity of the pset handling could be expressed using
    // ifValue and addNode but it might result in present configs failing
    edm::ParameterSetDescription ps;
    ps.add<edm::InputTag>("src");
    ps.addOptional<double>("jecUncertaintyValue");
    ps.addOptional<std::string>("jetCorrUncertaintyTag")->setComment("only used if 'jecUncertaintyValue' not declared");
    ps.addOptional<edm::FileInPath>("jetCorrInputFileName")
        ->setComment("only used if 'jecUncertaintyValue' not declared");
    ps.addOptional<std::string>("jetCorrPayloadName")
        ->setComment("only used if neither 'jecUncertaintyValue' nor 'jetCorrInputFileName' are declared");

    ps.add<bool>("addResidualJES");
    ps.addOptional<edm::InputTag>("jetCorrLabelUpToL3")->setComment("only used of 'addResidualJES' is set true");
    ps.addOptional<edm::InputTag>("jetCorrLabelUpToL3Res")->setComment("only used of 'addResidualJES' is set true");

    ps.add<double>("jetCorrEtaMax", 9.9);
    ps.add<double>("shiftBy");
    ps.addUntracked<int>("verbosity", 0);
    iDesc.addDefault(ps);
  }

private:
  void produce(edm::Event& evt, const edm::EventSetup& es) override {
    if (verbosity_) {
      std::cout << "<ShiftedJetProducerT::produce>:" << std::endl;
      std::cout << " moduleLabel = " << moduleLabel_ << std::endl;
      std::cout << " src = " << src_.label() << std::endl;
    }

    JetCollection const& originalJets = evt.get(srcToken_);

    edm::Handle<reco::JetCorrector> jetCorrUpToL3;
    edm::Handle<reco::JetCorrector> jetCorrUpToL3Res;
    if (evt.isRealData() && addResidualJES_) {
      jetCorrUpToL3 = evt.getHandle(jetCorrTokenUpToL3_);
      jetCorrUpToL3Res = evt.getHandle(jetCorrTokenUpToL3Res_);
    }

    if (not jecUncertaintyValue_) {
      if (jetCorrPayloadToken_.isInitialized()) {
        auto cacheID = es.get<JetCorrectionsRecord>().cacheIdentifier();
        if (cacheID != recordIdentifier_) {
          const JetCorrectorParametersCollection& jetCorrParameterSet = es.getData(jetCorrPayloadToken_);
          const JetCorrectorParameters& jetCorrParameters = (jetCorrParameterSet)[jetCorrUncertaintyTag_];
          jecUncertainty_.emplace(jetCorrParameters);
          recordIdentifier_ = cacheID;
        }
      }
    }

    JetCollection shiftedJets;
    shiftedJets.reserve(originalJets.size());
    for (auto const& originalJet : originalJets) {
      reco::Candidate::LorentzVector originalJetP4 = originalJet.p4();
      if (verbosity_) {
        std::cout << "originalJet: Pt = " << originalJetP4.pt() << ", eta = " << originalJetP4.eta()
                  << ", phi = " << originalJetP4.phi() << std::endl;
      }

      double shift = 0.;
      if (jecUncertaintyValue_) {
        shift = *jecUncertaintyValue_;
      } else {
        jecUncertainty_->setJetEta(originalJetP4.eta());
        jecUncertainty_->setJetPt(originalJetP4.pt());

        shift = jecUncertainty_->getUncertainty(true);
      }
      if (verbosity_) {
        std::cout << "shift = " << shift << std::endl;
      }

      if (evt.isRealData() && addResidualJES_) {
        reco::Candidate::LorentzVector rawJetP4 = pat::RawJetExtractorT<T>{}(originalJet);
        if (rawJetP4.E() > 1.e-1) {
          reco::Candidate::LorentzVector corrJetP4upToL3 =
              jetCorrExtractor_(originalJet, jetCorrUpToL3.product(), jetCorrEtaMax_, &rawJetP4);
          reco::Candidate::LorentzVector corrJetP4upToL3Res =
              jetCorrExtractor_(originalJet, jetCorrUpToL3Res.product(), jetCorrEtaMax_, &rawJetP4);
          if (corrJetP4upToL3.E() > 1.e-1 && corrJetP4upToL3Res.E() > 1.e-1) {
            double residualJES = (corrJetP4upToL3Res.E() / corrJetP4upToL3.E()) - 1.;
            shift = sqrt(shift * shift + residualJES * residualJES);
          }
        }
      }

      shift *= shiftBy_;
      if (verbosity_) {
        std::cout << "shift*shiftBy = " << shift << std::endl;
      }

      shiftedJets.emplace_back(originalJet);
      shiftedJets.back().setP4((1. + shift) * originalJetP4);
      if (verbosity_) {
        auto const& shiftedJet = shiftedJets.back();
        std::cout << "shiftedJet: Pt = " << shiftedJet.pt() << ", eta = " << shiftedJet.eta()
                  << ", phi = " << shiftedJet.phi() << std::endl;
      }
    }

    evt.emplace(putToken_, std::move(shiftedJets));
  }

  std::string moduleLabel_;

  edm::InputTag src_;
  edm::EDGetTokenT<JetCollection> srcToken_;
  edm::EDPutTokenT<JetCollection> putToken_;

  edm::FileInPath jetCorrInputFileName_;
  std::string jetCorrPayloadName_;
  edm::ESGetToken<JetCorrectorParametersCollection, JetCorrectionsRecord> jetCorrPayloadToken_;
  std::string jetCorrUncertaintyTag_;
  std::optional<JetCorrectionUncertainty> jecUncertainty_;
  unsigned long long recordIdentifier_ = 0;

  bool addResidualJES_;
  edm::InputTag jetCorrLabelUpToL3_;                            // L1+L2+L3 correction
  edm::EDGetTokenT<reco::JetCorrector> jetCorrTokenUpToL3_;     // L1+L2+L3 correction
  edm::InputTag jetCorrLabelUpToL3Res_;                         // L1+L2+L3+Residual correction
  edm::EDGetTokenT<reco::JetCorrector> jetCorrTokenUpToL3Res_;  // L1+L2+L3+Residual correction
  double jetCorrEtaMax_;  // do not use JEC factors for |eta| above this threshold (recommended default = 4.7),
                          // in order to work around problem with CMSSW_4_2_x JEC factors at high eta,
                          // reported in
                          //  https://hypernews.cern.ch/HyperNews/CMS/get/jes/270.html
                          //  https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/1259/1.html
  Textractor jetCorrExtractor_;

  std::optional<double> jecUncertaintyValue_;

  double shiftBy_;  // set to +1.0/-1.0 for up/down variation of energy scale

  int verbosity_;  // flag to enabled/disable debug output
};

#endif
