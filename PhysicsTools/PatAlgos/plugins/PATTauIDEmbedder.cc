#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/PATTauDiscriminator.h"
#include "DataFormats/TauReco/interface/TauDiscriminatorContainer.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "FWCore/Utilities/interface/transform.h"

class PATTauIDEmbedder : public edm::stream::EDProducer<> {
public:
  explicit PATTauIDEmbedder(const edm::ParameterSet&);
  ~PATTauIDEmbedder() override{};

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  //--- configuration parameters
  edm::EDGetTokenT<pat::TauCollection> src_;
  typedef std::pair<std::string, edm::InputTag> NameTag;
  typedef std::pair<std::string, int> NameWPIdx;
  //to save input module tag and corresponding pairs <working point name for the output tree, WP index in the input ID container>
  typedef std::pair<edm::InputTag, std::vector<NameWPIdx> > IDContainerData;
  std::vector<NameTag> tauIDSrcs_;
  std::vector<std::vector<NameWPIdx> > tauIDSrcContainers_;
  std::vector<edm::EDGetTokenT<pat::PATTauDiscriminator> > patTauIDTokens_;
  std::vector<edm::EDGetTokenT<reco::TauDiscriminatorContainer> > patTauIDContainerTokens_;
  size_t nNewPlainTauIds_;
  size_t nNewTauIds_;
};

PATTauIDEmbedder::PATTauIDEmbedder(const edm::ParameterSet& cfg) {
  src_ = consumes<pat::TauCollection>(cfg.getParameter<edm::InputTag>("src"));
  // read the different tau ID names
  edm::ParameterSet idps = cfg.getParameter<edm::ParameterSet>("tauIDSources");
  std::vector<std::string> names = idps.getParameterNamesForType<edm::ParameterSet>();
  std::map<std::string, IDContainerData> idContainerMap;
  for (auto const& name : names) {
    edm::ParameterSet idp = idps.getParameter<edm::ParameterSet>(name);
    int wpidx = idp.getParameter<int>("workingPointIndex");
    edm::InputTag tag = idp.getParameter<edm::InputTag>("inputTag");
    if (wpidx == -99) {
      tauIDSrcs_.push_back(NameTag(name, tag));
    } else {
      std::map<std::string, IDContainerData>::iterator it;
      it = idContainerMap.insert({tag.label() + tag.instance(), {tag, std::vector<NameWPIdx>()}}).first;
      it->second.second.push_back(NameWPIdx(name, wpidx));
    }
  }
  // but in any case at least once
  if (tauIDSrcs_.empty() && idContainerMap.empty())
    throw cms::Exception("Configuration") << "PATTauProducer: id addTauID is true, you must specify:\n"
                                          << "\tPSet tauIDSources = { \n"
                                          << "\t\tInputTag <someName> = <someTag>   // as many as you want \n "
                                          << "\t}\n";
  for (auto const& mapEntry : idContainerMap) {
    tauIDSrcContainers_.push_back(mapEntry.second.second);
    patTauIDContainerTokens_.push_back(mayConsume<reco::TauDiscriminatorContainer>(mapEntry.second.first));
  }
  patTauIDTokens_ = edm::vector_transform(
      tauIDSrcs_, [this](NameTag const& tag) { return mayConsume<pat::PATTauDiscriminator>(tag.second); });
  nNewPlainTauIds_ = tauIDSrcs_.size();
  nNewTauIds_ = nNewPlainTauIds_;
  for (auto const& cont : tauIDSrcContainers_) {
    nNewTauIds_ += cont.size();
  }

  produces<std::vector<pat::Tau> >();
}

void PATTauIDEmbedder::produce(edm::Event& evt, const edm::EventSetup& es) {
  edm::Handle<pat::TauCollection> inputTaus;
  evt.getByToken(src_, inputTaus);

  auto outputTaus = std::make_unique<std::vector<pat::Tau> >();
  outputTaus->reserve(inputTaus->size());

  int tau_idx = 0;
  for (pat::TauCollection::const_iterator inputTau = inputTaus->begin(); inputTau != inputTaus->end();
       ++inputTau, ++tau_idx) {
    pat::Tau outputTau(*inputTau);
    pat::TauRef inputTauRef(inputTaus, tau_idx);
    size_t nTauIds = inputTau->tauIDs().size();
    std::vector<pat::Tau::IdPair> tauIds(nTauIds + nNewTauIds_);

    // copy IDs that are already stored in PAT taus
    for (size_t i = 0; i < nTauIds; ++i) {
      tauIds[i] = inputTau->tauIDs().at(i);
    }

    // store IDs that were produced in PATTauDiscriminator format
    edm::Handle<pat::PATTauDiscriminator> tauDiscr;
    for (size_t i = 0; i < nNewPlainTauIds_; ++i) {
      evt.getByToken(patTauIDTokens_[i], tauDiscr);
      tauIds[nTauIds + i].first = tauIDSrcs_[i].first;
      tauIds[nTauIds + i].second = (*tauDiscr)[inputTauRef];
    }

    // store IDs that were produced in PATTauDiscriminatorContainer format
    size_t nEmbeddedIDs = nTauIds + nNewPlainTauIds_;
    for (size_t i = 0; i < tauIDSrcContainers_.size(); ++i) {
      auto const& tauDiscrCont = evt.get(patTauIDContainerTokens_[i]);
      for (size_t j = 0; j < tauIDSrcContainers_[i].size(); ++j) {
        tauIds[nEmbeddedIDs + j].first = tauIDSrcContainers_[i][j].first;
        int wpIdx = tauIDSrcContainers_[i][j].second;
        if (wpIdx < 0) {
          if (tauDiscrCont[inputTauRef].rawValues.size() == 1)
            //Only 0th component filled with default value if prediscriminor in PatTauDiscriminator failed.
            tauIds[nEmbeddedIDs + j].second = tauDiscrCont[inputTauRef].rawValues.at(0);
          else
            //uses negative indices to access rawValues. In most cases only one rawValue at WPIdx=-1 exists.
            tauIds[nEmbeddedIDs + j].second = tauDiscrCont[inputTauRef].rawValues.at(-1 - wpIdx);
        } else {
          if (tauDiscrCont[inputTauRef].workingPoints.empty())
            //WP vector not filled if prediscriminor in PatTauDiscriminator failed. Set PAT output to false in this case
            tauIds[nEmbeddedIDs + j].second = 0.0;
          else
            tauIds[nEmbeddedIDs + j].second = tauDiscrCont[inputTauRef].workingPoints.at(wpIdx);
        }
      }
      nEmbeddedIDs += tauIDSrcContainers_[i].size();
    }

    outputTau.setTauIDs(tauIds);
    outputTaus->push_back(outputTau);
  }

  evt.put(std::move(outputTaus));
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATTauIDEmbedder);
