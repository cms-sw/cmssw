#ifndef GSGsfTrackCandidateMaker_h
#define GSGsfTrackCandidateMaker_h

//
// Package:    EgammaElectronAlgos
// Class:      GSGsfTrackCandidateMaker
// 
//
// Description: Produce TrackCandidates for GSF fitting from seeds
// and write Associationmap trackcandidates-seeds at the same time
//
//
// Original Author:  Patrick Janot 
//

#include "FWCore/Framework/interface/EDProducer.h"

class TrackerGeometry;

namespace edm { 
  class Event;
  class EventSetup;
  class ParameterSet;
}

class GSGsfTrackCandidateMaker : public edm::EDProducer
{
 public:
  
  explicit GSGsfTrackCandidateMaker(const edm::ParameterSet& conf);
  
  virtual ~GSGsfTrackCandidateMaker();
  
  virtual void beginJob (const edm::EventSetup& es);
  
  virtual void produce(edm::Event& e, const edm::EventSetup& es);
  
 private:
  
  const TrackerGeometry* theTrackerGeometry;
  
  std::string seedProducer; 
  std::string seedLabel; 
  bool rejectOverlaps;
  double ptCut;
  unsigned int minimumNumberOfHits;


};

#endif
