#ifndef Calibration_IsolatedEcalPixelTrackCandidateProducer_h
#define Calibration_IsolatedEcalPixelTrackCandidateProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

//
// class decleration
//

class IsolatedEcalPixelTrackCandidateProducer : public edm::global::EDProducer<> {

public:
  explicit IsolatedEcalPixelTrackCandidateProducer(const edm::ParameterSet&);
  ~IsolatedEcalPixelTrackCandidateProducer();

private:
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  const edm::EDGetTokenT<EcalRecHitCollection> tok_ee;
  const edm::EDGetTokenT<EcalRecHitCollection> tok_eb;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tok_trigcand;
  const double coneSizeEta0_;
  const double coneSizeEta1_;
  const double hitCountEthr_;
  const double hitEthr_;
};

#endif
