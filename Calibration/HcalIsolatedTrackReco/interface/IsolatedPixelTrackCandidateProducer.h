#ifndef Calibration_IsolatedPixelTrackCandidateProducer_h
#define Calibration_IsolatedPixelTrackCandidateProducer_h

/* \class IsolatedPixelTrackCandidateProducer
 *
 *  
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

class IsolatedPixelTrackCandidateProducer : public edm::EDProducer {

 public:

  IsolatedPixelTrackCandidateProducer (const edm::ParameterSet& ps);
  ~IsolatedPixelTrackCandidateProducer();


  virtual void beginJob (edm::EventSetup const & es){};
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

 private:

  edm::InputTag l1eTauJetsSource_;
  edm::InputTag pixelTracksSource_;
  double pixelIsolationConeSize_;
};


#endif
