#ifndef Calibration_IsolatedEcalPixelTrackCandidateProducer_h
#define Calibration_IsolatedEcalPixelTrackCandidateProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

//
// class decleration
//

class IsolatedEcalPixelTrackCandidateProducer : public edm::EDProducer {

public:
  explicit IsolatedEcalPixelTrackCandidateProducer(const edm::ParameterSet&);
  ~IsolatedEcalPixelTrackCandidateProducer();

private:

  virtual void beginJob() ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  double coneSizeEta0_, coneSizeEta1_;
  double hitCountEthr_;
  double hitEthr_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_ee;
  edm::EDGetTokenT<EcalRecHitCollection> tok_eb;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tok_trigcand;
};

#endif
