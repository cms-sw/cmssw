// Author: Felice Pantaleo, Leonardo Cristella - felice.pantaleo@cern.ch, leonardo.cristella@cern.ch
// Date: 09/2021
// Refactored : Theo Cuisset - theo.cuisset@polytechnique.edu

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/HGCalReco/interface/TICLCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/SimCluster.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimTrackster.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimTracksterFwd.h"

#include "DataFormats/HGCalReco/interface/Common.h"

#include <vector>

using namespace ticl;

/**
* @brief Produces TICLCandidates using MC truth
* Uses two sim collections (each with the Calorimeter truth object as well as corresponding tracksters) : the "base" collections (usually CaloParticle) and "sub" collections (typically SimCluster from crossedBoundary SimTracks)
* TICLCandidate recoTrack pointers are set from "base" SimTrackster  matched reco tracks.
* MTD time is set to MtdSimTrackster time (matched by shared SimTrack to "base" sim object)
* 4-momentum for neutral is set from "base" simTrackster energy (for charged from sim true energy & reco track direction)
* 
* There is a 1-1 mapping between SimTICLCandidate and "base" SimTrackster
*/
template <typename BaseSimObject_t, typename SubSimObject_t>
class SimTICLCandidateProducerT : public edm::global::EDProducer<> {
public:
  explicit SimTICLCandidateProducerT(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<std::vector<BaseSimObject_t>> baseCaloSimObjects_token_;
  const edm::EDGetTokenT<std::vector<SubSimObject_t>> subCaloSimObjects_token_;
  const edm::EDGetTokenT<edm::RefVector<std::vector<BaseSimObject_t>>> subToBaseSimObject_map_token_;

  const edm::EDGetTokenT<std::vector<Trackster>> baseSimTracksters_token_;
  const edm::EDGetTokenT<std::vector<Trackster>> subSimTracksters_token_;
  const edm::EDGetTokenT<edm::RefVector<std::vector<BaseSimObject_t>>> baseSimTracksterToBaseSimObject_map_token_;
  const edm::EDGetTokenT<edm::RefVector<std::vector<SubSimObject_t>>> subSimTracksterToSubSimObject_map_token_;
  const edm::EDGetTokenT<CaloParticleRefVector> baseSimTracksterToCaloParticle_map_token_;

  const edm::EDGetTokenT<MtdSimTracksterCollection> MTDSimTrackstersToken_;
  const edm::EDGetTokenT<std::vector<reco::Track>> recoTracksToken_;

  const edm::EDPutTokenT<std::vector<TICLCandidate>> SimTICLCandidates_token_;
};

template class SimTICLCandidateProducerT<CaloParticle, SimCluster>;
template class SimTICLCandidateProducerT<SimCluster, SimCluster>;

using SimTICLCandidateProducerUsingCaloParticle = SimTICLCandidateProducerT<CaloParticle, SimCluster>;
DEFINE_FWK_MODULE(SimTICLCandidateProducerUsingCaloParticle);
using SimTICLCandidateProducerUsingSimCluster = SimTICLCandidateProducerT<SimCluster, SimCluster>;
DEFINE_FWK_MODULE(SimTICLCandidateProducerUsingSimCluster);

template <typename BaseSimObject_t, typename SubSimObject_t>
SimTICLCandidateProducerT<BaseSimObject_t, SubSimObject_t>::SimTICLCandidateProducerT(const edm::ParameterSet& ps)
    : baseCaloSimObjects_token_(consumes(ps.getParameter<edm::InputTag>("baseCaloSimObjects"))),
      subCaloSimObjects_token_(consumes(ps.getParameter<edm::InputTag>("subCaloSimObjects"))),
      subToBaseSimObject_map_token_(consumes(ps.getParameter<edm::InputTag>("subToBaseMap"))),

      baseSimTracksters_token_(consumes(ps.getParameter<edm::InputTag>("baseSimTracksters"))),
      subSimTracksters_token_(consumes(ps.getParameter<edm::InputTag>("subSimTracksters"))),

      baseSimTracksterToBaseSimObject_map_token_(
          consumes(ps.getParameter<edm::InputTag>("baseSimTracksterToBaseSimObject_map"))),
      subSimTracksterToSubSimObject_map_token_(
          consumes(ps.getParameter<edm::InputTag>("subSimTracksterToSubSimObject_map"))),
      baseSimTracksterToCaloParticle_map_token_(
          consumes(ps.getParameter<edm::InputTag>("baseSimTracksterToCaloParticle_map"))),

      MTDSimTrackstersToken_(consumes<MtdSimTracksterCollection>(ps.getParameter<edm::InputTag>("MtdSimTracksters"))),
      recoTracksToken_(consumes<std::vector<reco::Track>>(ps.getParameter<edm::InputTag>("recoTracks"))),

      SimTICLCandidates_token_(produces<std::vector<TICLCandidate>>()) {}

template <typename BaseSimObject_t, typename SubSimObject_t>
void SimTICLCandidateProducerT<BaseSimObject_t, SubSimObject_t>::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("baseCaloSimObjects", edm::InputTag("mix", "MergedCaloTruth"))
      ->setComment(
          "Base sim object collection (=CaloParticle) as SimCluster dataformat (one base sim object = one "
          "SimTICLCandidate)");
  desc.add<edm::InputTag>("subCaloSimObjects", edm::InputTag("mix", "MergedCaloTruth"))
      ->setComment("'Sub' calo sim object collection. Currently unused.");
  desc.add<edm::InputTag>("subToBaseMap", edm::InputTag("mix", "MergedCaloTruth"))
      ->setComment(
          "Map (edm::RefVector) from 'subCaloSimObjects' collection to 'baseCaloSimObjects' collections (ie SimCluster "
          "-> CaloParticle map)");

  desc.add<edm::InputTag>("baseSimTracksters")
      ->setComment(
          "'Base' trackster collection (typically SimTrackster from CaloParticle). Used for adding reco tracks to "
          "TICLCandidate. One base simTracksters makes one SimTICLCandidate");
  desc.add<edm::InputTag>("baseSimTracksterToBaseSimObject_map")
      ->setComment("Map 'baseSimTracksters' collection -> 'baseCaloSimObjects' collection ");
  desc.add<edm::InputTag>("baseSimTracksterToCaloParticle_map")
      ->setComment(
          "Map 'baseSimTracksters' collection -> CaloParticle (used only to get the vertex time of CaloParticle)");

  desc.add<edm::InputTag>("subSimTracksters")
      ->setComment(
          "'Sub' trackster collection : tracksters whose references are added to TICLCandidate object (usually "
          "SimTrackster from boundary SimCluster)");
  desc.add<edm::InputTag>("subSimTracksterToSubSimObject_map");

  desc.add<edm::InputTag>("MtdSimTracksters", edm::InputTag("mix", "MergedMtdTruthST"));
  desc.add<edm::InputTag>("recoTracks", edm::InputTag("generalTracks"));

  descriptions.addWithDefaultLabel(desc);
}

template <typename BaseSimObject_t, typename SubSimObject_t>
void SimTICLCandidateProducerT<BaseSimObject_t, SubSimObject_t>::produce(edm::StreamID,
                                                                         edm::Event& evt,
                                                                         const edm::EventSetup& es) const {
  edm::Handle<std::vector<BaseSimObject_t>> baseCaloSimObjects_h = evt.getHandle(baseCaloSimObjects_token_);
  // not needed currently but kept in case of future need
  edm::Handle<std::vector<SubSimObject_t>> subCaloSimObjects_h = evt.getHandle(subCaloSimObjects_token_);

  edm::RefVector<std::vector<BaseSimObject_t>> const& subToBaseSimObject_map = evt.get(subToBaseSimObject_map_token_);
  edm::RefVector<std::vector<BaseSimObject_t>> const& baseSimTracksterToBaseSimObject_map =
      evt.get(baseSimTracksterToBaseSimObject_map_token_);
  edm::RefVector<std::vector<SubSimObject_t>> const& subSimTracksterToSubSimObject_map =
      evt.get(subSimTracksterToSubSimObject_map_token_);
  CaloParticleRefVector const& baseSimTracksterToCaloParticle_map = evt.get(baseSimTracksterToCaloParticle_map_token_);

  if (!baseCaloSimObjects_h.isValid() || !baseSimTracksterToBaseSimObject_map.id().isValid()) {
    edm::LogWarning("SimTICLCandidateProducer") << "Missing input collections. Producing empty outputs.";
    evt.emplace(SimTICLCandidates_token_);
    return;
  }

  TracksterCollection const& baseSimTracksters = evt.get(baseSimTracksters_token_);
  // Need handle for TICLCandidate::addTrackster
  edm::Handle<TracksterCollection> subSimTracksters_h = evt.getHandle(subSimTracksters_token_);
  TracksterCollection const& subSimTracksters = *subSimTracksters_h;

  assert(subToBaseSimObject_map.id() == baseCaloSimObjects_h.id());
  assert(subToBaseSimObject_map.size() == subCaloSimObjects_h->size());
  assert(baseSimTracksterToBaseSimObject_map.id() == baseCaloSimObjects_h.id());
  assert(baseSimTracksterToBaseSimObject_map.size() == baseSimTracksters.size());
  assert(subSimTracksterToSubSimObject_map.id() == subCaloSimObjects_h.id());
  assert(subSimTracksterToSubSimObject_map.size() == subSimTracksters.size());

  auto result_ticlCandidates = std::vector<TICLCandidate>(baseSimTracksters.size());

  auto mapBaseSimTracksterToBaseSimObject =
      [&](TracksterCollection::size_type baseSimTs_i) -> edm::Ref<std::vector<BaseSimObject_t>> {
    return baseSimTracksterToBaseSimObject_map[baseSimTs_i];
  };  // returns edm::Ref<BaseSimObject_t>
  auto mapBaseSimTsToTICLCandidate = [&](TracksterCollection::size_type baseSimTs_i) -> TICLCandidate& {
    return result_ticlCandidates[baseSimTs_i];
  };  // 1-1 mapping (but kept as a function in case it needs to be changed)
  auto mapTICLCandidateToBaseSimTsIndex =
      [&](TracksterCollection::size_type ticlCand_i) -> TracksterCollection::size_type {
    return ticlCand_i;
  };  // 1-1 mapping (but kept as a function in case it needs to be changed)
  auto mapTICLCandidateToBaseSimObject = [&](TracksterCollection::size_type ticlCand_i) -> BaseSimObject_t const& {
    return *baseSimTracksterToBaseSimObject_map[ticlCand_i];
  };

  // Map from "base" SimObject to the SimTrackster corresponding. Reverse the RefVector. There is no check that the SimObject actually has a SimTrackster (in that case it will return an out of range index)
  auto mapBaseSimObjectToBaseSimTs = [&](std::size_t baseSimObject_i) -> TracksterCollection::size_type {
    return std::ranges::find_if(baseSimTracksterToBaseSimObject_map,
                                [baseSimObject_i](auto const& elem) { return elem.key() == baseSimObject_i; }) -
           baseSimTracksterToBaseSimObject_map.begin();
  };
  auto mapSubSimTsToTICLCandidate = [&](TracksterCollection::size_type subSimTs_i) -> TICLCandidate& {
    return mapBaseSimTsToTICLCandidate(
        mapBaseSimObjectToBaseSimTs(subToBaseSimObject_map[subSimTracksterToSubSimObject_map[subSimTs_i].key()].key()));
  };  // return TICLCandidate&

  MtdSimTracksterCollection const& MTDSimTracksters = evt.get(MTDSimTrackstersToken_);

  edm::Handle<std::vector<reco::Track>> recoTracks_h = evt.getHandle(recoTracksToken_);

  // map between simTrack and Mtd SimTracksters to loop on them only one
  std::unordered_map<unsigned int, const MtdSimTrackster*> SimTrackToMtdST;
  for (unsigned int i = 0; i < MTDSimTracksters.size(); ++i) {
    const auto& simTrack = MTDSimTracksters[i].g4Tracks()[0];
    SimTrackToMtdST[simTrack.trackId()] = &(MTDSimTracksters[i]);
  }

  /* ----- Step 1 :  Add tracksters from "sub" collection */
  for (size_t subSimTrackster_i = 0; subSimTrackster_i < subSimTracksters.size(); ++subSimTrackster_i) {
    TICLCandidate& cand = mapSubSimTsToTICLCandidate(subSimTrackster_i);
    cand.addTrackster(edm::Ptr<Trackster>(subSimTracksters_h, subSimTrackster_i));
  }

  /* ----- Step 2 :  Add tracks from "base" collection of tracksters */
  for (size_t baseSimTrackster_i = 0; baseSimTrackster_i < baseSimTracksters.size(); ++baseSimTrackster_i) {
    ticl::Trackster const& baseSimTrackster = baseSimTracksters[baseSimTrackster_i];
    BaseSimObject_t const& baseSimObject = *mapBaseSimTracksterToBaseSimObject(baseSimTrackster_i);

    if (!baseSimTrackster.vertices().empty()) {
      auto trackIndices = baseSimTrackster.trackIdxs();

      TICLCandidate& cand = mapBaseSimTsToTICLCandidate(baseSimTrackster_i);
      if (baseSimObject.charge() != 0) {
        for (const auto trackIndex : trackIndices) {
          cand.addTrackPtr(edm::Ptr<reco::Track>(recoTracks_h, trackIndex));
        }
      }
    }
  }

  /* ----- Step 3 :  Set pdgId, time, momentum
  Time : from BaseSimObject simTime
  MtdTime : from MtdSimTrackster built from the SimTrack of the genParticle (probably this needs to be clarified, not clear what happens if that simtrack never reaches MTD)
  Then : 
   * if there is a matched reco track to SimTICLCandidate (= if base sim object is charged && there is a recotrack matched to TrackingParticle corresponding to first SimTrack, cf SimTrackstersProducer)
      - energies : from subSimTracksters energies (momentum at boundary of SimTracks)
      - PdgID : from baseSimObject
      - charge : from baseSimObject
   * if there is no matched reco track (= neutral at sim level or charged where track is not recoed good enough)
      - energies : from baseSimTrackster energy (= momentum at boundary if baseSimObj SimTrack crossed boundary, otherwise SimObject energy)
      - pdgId : from baseSimObject
      - charge : always 0 (even if baseSimObject was charged, in this case check pdgId to get "sim" charge)
   */
  auto isHad = [](int pdgId) {
    pdgId = std::abs(pdgId);
    if (pdgId == 111)
      return false;
    return (pdgId > 100 and pdgId < 900) or (pdgId > 1000 and pdgId < 9000);
  };

  for (size_t i = 0; i < result_ticlCandidates.size(); ++i) {
    BaseSimObject_t const& baseSimObject = mapTICLCandidateToBaseSimObject(i);
    TICLCandidate& cand = result_ticlCandidates[i];
    float rawEnergy = 0.f;
    float regressedEnergy = 0.f;

    const auto& simTrack = baseSimObject.g4Tracks()[0];
    auto pos = SimTrackToMtdST.find(simTrack.trackId());
    if (pos != SimTrackToMtdST.end()) {
      auto MTDst = pos->second;
      // TODO: once the associators have been implemented check if the MTDst is associated with a reco before adding the MTD time
      cand.setMTDTime(MTDst->time(), 0);
    }

    // Time of SimVertex of genParticle
    cand.setTime(baseSimTracksterToCaloParticle_map[mapTICLCandidateToBaseSimTsIndex(i)]->simTime(), 0);

    for (const auto& trackster : cand.tracksters()) {
      rawEnergy += trackster->raw_energy();
      regressedEnergy += trackster->regressed_energy();
    }
    cand.setRawEnergy(rawEnergy);

    auto pdgId = baseSimObject.pdgId();
    auto charge = baseSimObject.charge();
    if (cand.trackPtr().isNonnull()) {
      auto const& track = cand.trackPtr().get();
      if (std::abs(pdgId) == 13) {
        cand.setPdgId(pdgId);
      } else {
        cand.setPdgId((isHad(pdgId) ? 211 : 11) * charge);
      }
      cand.setCharge(charge);
      math::XYZTLorentzVector p4(regressedEnergy * track->momentum().unit().x(),
                                 regressedEnergy * track->momentum().unit().y(),
                                 regressedEnergy * track->momentum().unit().z(),
                                 regressedEnergy);
      cand.setP4(p4);
    } else {  // neutral candidates
      // a neutral candidate with a charged CaloParticle is charged without a reco track associated with it
      // set the charge = 0, but keep the real pdgId to keep track of that
      if (charge != 0)
        cand.setPdgId(isHad(pdgId) ? 211 : 11);
      else if (pdgId == 111)
        cand.setPdgId(pdgId);
      else
        cand.setPdgId(isHad(pdgId) ? 130 : 22);
      cand.setCharge(0);

      auto particleType = tracksterParticleTypeFromPdgId(cand.pdgId(), 1);
      cand.setIdProbability(particleType, 1.f);

      const Trackster& baseSimTrackster = baseSimTracksters[mapTICLCandidateToBaseSimTsIndex(i)];
      float regressedEnergy = baseSimTrackster.regressed_energy();
      math::XYZTLorentzVector p4(regressedEnergy * baseSimTrackster.barycenter().unit().x(),
                                 regressedEnergy * baseSimTrackster.barycenter().unit().y(),
                                 regressedEnergy * baseSimTrackster.barycenter().unit().z(),
                                 regressedEnergy);
      cand.setP4(p4);
    }
  }

  evt.emplace(SimTICLCandidates_token_, std::move(result_ticlCandidates));
}
