#ifndef Calibration_IPTCorrector_h
#define Calibration_IPTCorrector_h

/* \class IsolatedPixelTrackCandidateProducer
 *
 *  
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Ref.h"

//#include "DataFormats/Common/interface/Provenance.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"

class IPTCorrector : public edm::EDProducer {

 public:

  IPTCorrector (const edm::ParameterSet& ps);
  ~IPTCorrector();

  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

 private:
	
  edm::InputTag corSource_;
  edm::InputTag uncorSource_;
  double assocCone_;
};


#endif
