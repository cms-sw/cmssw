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


/*
namespace edm {
  class EDProducer;
  class Event;
  class EventSetup;
  class ParameterSet;
  template<typename T> class Handle;
}
*/
namespace edm {
  template<typename T> class Handle;
}


class RecoTrackAccumulator : public DigiAccumulatorMixMod 
{
 public:
  explicit RecoTrackAccumulator(const edm::ParameterSet& conf, edm::EDProducer& mixMod);
  virtual ~RecoTrackAccumulator();
  
  virtual void initializeEvent(edm::Event const& e, edm::EventSetup const& c);
  virtual void accumulate(edm::Event const& e, edm::EventSetup const& c);
  virtual void accumulate(PileUpEventPrincipal const& e, edm::EventSetup const& c);
  virtual void finalizeEvent(edm::Event& e, edm::EventSetup const& c);
  
 private:
  std::auto_ptr<reco::TrackCollection> NewTrackList_;
  edm::InputTag GeneralTrackInput_;
  std::string GeneralTrackOutput_;
};


#endif
