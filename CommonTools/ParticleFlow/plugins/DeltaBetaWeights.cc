// Weight for neutral particles based on distance with charged
//
// Original Author:  Michail Bachtis,40 1-B08,+41227678176,
//         Created:  Mon Dec  9 13:18:05 CET 2013
//
// edited by Pavel Jez
//

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <memory>

class DeltaBetaWeights : public edm::global::EDProducer<> {
public:
  explicit DeltaBetaWeights(const edm::ParameterSet&);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  // ----------member data ---------------------------
  edm::InputTag src_;
  edm::InputTag pfCharged_;
  edm::InputTag pfPU_;

  edm::EDGetTokenT<edm::View<reco::Candidate> > pfCharged_token;
  edm::EDGetTokenT<edm::View<reco::Candidate> > pfPU_token;
  edm::EDGetTokenT<edm::View<reco::Candidate> > src_token;
};

DeltaBetaWeights::DeltaBetaWeights(const edm::ParameterSet& iConfig)
    : src_(iConfig.getParameter<edm::InputTag>("src")),
      pfCharged_(iConfig.getParameter<edm::InputTag>("chargedFromPV")),
      pfPU_(iConfig.getParameter<edm::InputTag>("chargedFromPU")) {
  produces<reco::PFCandidateCollection>();

  pfCharged_token = consumes<edm::View<reco::Candidate> >(pfCharged_);
  pfPU_token = consumes<edm::View<reco::Candidate> >(pfPU_);
  src_token = consumes<edm::View<reco::Candidate> >(src_);

  // pfCharged_token = consumes<reco::PFCandidateCollection>(pfCharged_);
  // pfPU_token = consumes<reco::PFCandidateCollection>(pfPU_);
  // src_token = consumes<reco::PFCandidateCollection>(src_);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(DeltaBetaWeights);

// ------------ method called to produce the data  ------------
void DeltaBetaWeights::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  using namespace edm;

  edm::View<reco::Candidate> const pfCharged = iEvent.get(pfCharged_token);
  edm::View<reco::Candidate> const& pfPU = iEvent.get(pfPU_token);
  edm::View<reco::Candidate> const& src = iEvent.get(src_token);

  double sumNPU = .0;
  double sumPU = .0;

  std::unique_ptr<reco::PFCandidateCollection> out(new reco::PFCandidateCollection);

  out->reserve(src.size());
  for (const reco::Candidate& cand : src) {
    if (cand.charge() != 0) {
      // this part of code should be executed only if input collection is not entirely composed of neutral candidates, i.e. never by default
      edm::LogWarning("DeltaBetaWeights")
          << "Trying to reweight charged particle... saving it to output collection without any change";
      out->emplace_back(cand.charge(), cand.p4(), reco::PFCandidate::ParticleType::X);
      (out->back()).setParticleType((out->back()).translatePdgIdToType(cand.pdgId()));
      continue;
    }

    sumNPU = 1.0;
    sumPU = 1.0;
    double eta = cand.eta();
    double phi = cand.phi();
    for (const reco::Candidate& chCand : pfCharged) {
      double sum = (chCand.pt() * chCand.pt()) / (deltaR2(eta, phi, chCand.eta(), chCand.phi()));
      if (sum > 1.0)
        sumNPU *= sum;
    }
    sumNPU = 0.5 * log(sumNPU);

    for (const reco::Candidate& puCand : pfPU) {
      double sum = (puCand.pt() * puCand.pt()) / (deltaR2(eta, phi, puCand.eta(), puCand.phi()));
      if (sum > 1.0)
        sumPU *= sum;
    }
    sumPU = 0.5 * log(sumPU);

    reco::PFCandidate neutral = reco::PFCandidate(cand.charge(), cand.p4(), reco::PFCandidate::ParticleType::X);
    neutral.setParticleType(neutral.translatePdgIdToType(cand.pdgId()));
    if (sumNPU + sumPU > 0)
      neutral.setP4(((sumNPU) / (sumNPU + sumPU)) * neutral.p4());
    out->push_back(neutral);
  }

  iEvent.put(std::move(out));
}
