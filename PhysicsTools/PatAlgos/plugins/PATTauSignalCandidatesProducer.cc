#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"  //MB: can use CompositePtrCandidate, but dictionaries not defined
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

class PATTauSignalCandidatesProducer : public edm::stream::EDProducer<> {
public:
  explicit PATTauSignalCandidatesProducer(const edm::ParameterSet&);
  ~PATTauSignalCandidatesProducer() override{};

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  //--- configuration parameters
  edm::EDGetTokenT<pat::TauCollection> tausToken_;
  const bool storeLostTracks_;
};

PATTauSignalCandidatesProducer::PATTauSignalCandidatesProducer(const edm::ParameterSet& cfg)
    : tausToken_(consumes<pat::TauCollection>(cfg.getParameter<edm::InputTag>("src"))),
      storeLostTracks_(cfg.getParameter<bool>("storeLostTracks")) {
  produces<std::vector<reco::VertexCompositePtrCandidate>>();
}

void PATTauSignalCandidatesProducer::produce(edm::Event& evt, const edm::EventSetup& es) {
  // Get the vector of taus
  edm::Handle<pat::TauCollection> inputTaus;
  evt.getByToken(tausToken_, inputTaus);

  auto outputCands = std::make_unique<std::vector<reco::VertexCompositePtrCandidate>>();
  outputCands->reserve(inputTaus->size() * 3);  //avarage number of tau signal cands
  for (size_t iTau = 0; iTau < inputTaus->size(); ++iTau) {
    for (const auto& cand : (*inputTaus)[iTau].signalCands()) {
      reco::VertexCompositePtrCandidate outCand(*cand);
      outCand.setStatus(iTau);  //trick to store index of the mother tau to be used in NanoAOD
      outCand.addDaughter(cand);
      outputCands->push_back(outCand);
    }
    if (storeLostTracks_) {
      for (const auto& cand : (*inputTaus)[iTau].signalLostTracks()) {
        reco::VertexCompositePtrCandidate outCand(*cand);
        outCand.setStatus(iTau);  //trick to store index of the mother tau to be used in NanoAOD
        auto pdgId = cand->pdgId();
        outCand.setPdgId(
            pdgId + 10000 * ((pdgId >= 0) -
                             (pdgId < 0)));  // increase abs(pdgId) by 10000 to distingish from "true" signal candidates
        outCand.addDaughter(cand);
        outputCands->push_back(outCand);
      }
    }
  }

  evt.put(std::move(outputCands));
}

void PATTauSignalCandidatesProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // patTauDecayCandidatesProducer
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag("slimmedTaus"));
  desc.add<bool>("storeLostTracks", true)
      ->setComment("If true, lostTracks will be stored together with other candidates with pdgId=+-10211");

  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PATTauSignalCandidatesProducer);
