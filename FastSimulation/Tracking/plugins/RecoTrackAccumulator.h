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
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimGeneral/MixingModule/interface/PileUpEventPrincipal.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"


namespace edm {
  class ConsumesCollector;
  template<typename T> class Handle;
  namespace one {
    class EDProducerBase;
  }
  class StreamID;
}


class RecoTrackAccumulator : public DigiAccumulatorMixMod 
{
 public:
  explicit RecoTrackAccumulator(const edm::ParameterSet& conf, edm::one::EDProducerBase& mixMod, edm::ConsumesCollector& iC);
  virtual ~RecoTrackAccumulator();
  
  virtual void initializeEvent(edm::Event const& e, edm::EventSetup const& c);
  virtual void accumulate(edm::Event const& e, edm::EventSetup const& c);
  virtual void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c, edm::StreamID const&) override;
  virtual void finalizeEvent(edm::Event& e, edm::EventSetup const& c);
  
 private:
  template<class T> void accumulateEvent(const T& e, edm::EventSetup const& c,const edm::InputTag & label);

  std::auto_ptr<reco::TrackCollection> NewTrackList_;
  std::auto_ptr<reco::TrackExtraCollection> NewTrackExtraList_;
  std::auto_ptr<TrackingRecHitCollection> NewHitList_;

  reco::TrackExtraRefProd rTrackExtras;
  TrackingRecHitRefProd rHits;

  edm::InputTag InputSignal_;
  edm::InputTag InputPileUp_;

  std::string GeneralTrackOutput_;
  std::string HitOutput_;
  std::string GeneralTrackExtraOutput_;

};


#endif
