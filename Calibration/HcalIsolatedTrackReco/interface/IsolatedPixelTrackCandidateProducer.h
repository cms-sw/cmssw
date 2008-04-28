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

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"

//#include "DataFormats/Common/interface/Provenance.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"



class IsolatedPixelTrackCandidateProducer : public edm::EDProducer {

 public:

  IsolatedPixelTrackCandidateProducer (const edm::ParameterSet& ps);
  ~IsolatedPixelTrackCandidateProducer();


  virtual void beginJob (edm::EventSetup const & es){};
  virtual void produce(edm::Event& evt, const edm::EventSetup& es);

 private:
	
  edm::InputTag hltGTseedlabel_;
  edm::InputTag l1eTauJetsSource_;
  edm::InputTag pixelTracksSource_;
  edm::InputTag ecalFilterLabel_;
  edm::ParameterSet parameters;

  double pixelIsolationConeSize_;
  double tauAssocCone_;
  double tauUnbiasCone_;

};


#endif
