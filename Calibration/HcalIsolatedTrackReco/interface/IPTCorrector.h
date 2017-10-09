#ifndef Calibration_IPTCorrector_h
#define Calibration_IPTCorrector_h

/* \class IsolatedPixelTrackCandidateProducer
 *
 *  
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

class IPTCorrector : public edm::global::EDProducer<>
{
public:
  IPTCorrector (const edm::ParameterSet& ps);

  virtual void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

private:
  const edm::EDGetTokenT<reco::TrackCollection> tok_cor_;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tok_uncor_;
  const double assocCone_;
};


#endif
