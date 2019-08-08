// This producer converts a list of TICLCandidates to a list of PFCandidates.

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/View.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "DataFormats/HGCalReco/interface/TICLCandidate.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include <unordered_map>
#include <memory>

#include "CLHEP/Units/SystemOfUnits.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "FWCore/Utilities/interface/transform.h"

class PFTICLProducer : public edm::global::EDProducer<> {
public:
  PFTICLProducer(const edm::ParameterSet&);
  ~PFTICLProducer() override {}

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  // inputs
  const edm::EDGetTokenT<edm::View<ticl::TICLCandidate>> ticl_candidates_;
};

DEFINE_FWK_MODULE(PFTICLProducer);

PFTICLProducer::PFTICLProducer(const edm::ParameterSet& conf)
    : ticl_candidates_(consumes<edm::View<ticl::TICLCandidate>>(conf.getParameter<edm::InputTag>("ticlCandidateSrc"))) {
  produces<reco::PFCandidateCollection>();
}

void PFTICLProducer::produce(edm::StreamID, edm::Event& evt, const edm::EventSetup& es) const {
  //get TICLCandidates
  edm::Handle<edm::View<ticl::TICLCandidate>> ticl_cand_h;
  evt.getByToken(ticl_candidates_, ticl_cand_h);
  const auto ticl_candidates = *ticl_cand_h;

  auto candidates = std::make_unique<reco::PFCandidateCollection>();
  // in good particle flow fashion, start from the tracks and go out
  for (const auto& ticl_cand : ticl_candidates) {
    const auto absPdgId = std::abs(ticl_cand.pdgId());
    const auto charge = ticl_cand.charge();
    const auto four_mom = ticl_cand.p4();

    reco::PFCandidate::ParticleType part_type;
    switch (absPdgId) {
      case 11:
        part_type = reco::PFCandidate::e;
        break;
      case 13:
        part_type = reco::PFCandidate::mu;
        break;
      case 22:
        part_type = reco::PFCandidate::gamma;
        break;
      case 130:
        part_type = reco::PFCandidate::h0;
        break;
      default:
        part_type = reco::PFCandidate::h;
    }

    candidates->emplace_back(charge, four_mom, part_type);

    auto& candidate = candidates->back();
    candidate.setTrackRef(ticl_cand.track_ref());
    candidate.setTime(ticl_cand.time(), ticl_cand.time_error());
  }

  evt.put(std::move(candidates));
}