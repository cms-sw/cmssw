#ifndef RecoTrackAccumulator_h
#define RecoTrackAccumulator_h

/** \class RecoTrackAccumulator
 *
 * RecoTrackAccumulator accumulates generalTracks from the hard and the pileup events
 *
 * \author Andrea Giammanco
 *
 * \version   Mar 11 2013  
 *
 ************************************************************/

#include "SimGeneral/MixingModule/interface/DigiAccumulatorMixMod.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/ProducesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

namespace edm {
  class ConsumesCollector;
  template <typename T>
  class Handle;
  class StreamID;
}  // namespace edm

class RecoTrackAccumulator : public DigiAccumulatorMixMod {
public:
  explicit RecoTrackAccumulator(const edm::ParameterSet& conf, edm::ProducesCollector, edm::ConsumesCollector& iC);
  ~RecoTrackAccumulator() override;

  void initializeEvent(edm::Event const& e, edm::EventSetup const& c) override;
  void accumulate(edm::Event const& e, edm::EventSetup const& c) override;
  void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c, edm::StreamID const&) override;
  void finalizeEvent(edm::Event& e, edm::EventSetup const& c) override;

private:
  template <class T>
  void accumulateEvent(const T& e, edm::EventSetup const& c, const edm::InputTag& label);

  std::unique_ptr<reco::TrackCollection> newTracks_;
  std::unique_ptr<reco::TrackExtraCollection> newTrackExtras_;
  std::unique_ptr<TrackingRecHitCollection> newHits_;

  reco::TrackRefProd rNewTracks;
  reco::TrackExtraRefProd rNewTrackExtras;
  TrackingRecHitRefProd rNewHits;

  edm::InputTag signalTracksTag;
  edm::InputTag pileUpTracksTag;

  std::string outputLabel;
};

#endif
