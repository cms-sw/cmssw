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
  template<class T> void accumulateEvent(const T& e, edm::EventSetup const& c,const edm::InputTag & label,const edm::InputTag & MVALabel);

  std::auto_ptr<reco::TrackCollection>  newTracks_;
  std::auto_ptr<reco::TrackExtraCollection> newTrackExtras_;
  std::auto_ptr<TrackingRecHitCollection> newHits_;
  std::vector<float> newMVAVals_;

  reco::TrackRefProd rNewTracks;
  reco::TrackExtraRefProd rNewTrackExtras;
  TrackingRecHitRefProd rNewHits;

  edm::InputTag signalTracksTag;
  edm::InputTag signalMVAValuesTag;
  edm::InputTag pileUpTracksTag;
  edm::InputTag pileUpMVAValuesTag;

  std::string outputLabel;
  std::string MVAOutputLabel;
  
};


#endif
