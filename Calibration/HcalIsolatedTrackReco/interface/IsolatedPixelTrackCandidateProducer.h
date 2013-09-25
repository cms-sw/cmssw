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
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"

class IsolatedPixelTrackCandidateProducer : public edm::EDProducer {

 public:

  IsolatedPixelTrackCandidateProducer (const edm::ParameterSet& ps);
  ~IsolatedPixelTrackCandidateProducer();


  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
  virtual void produce(edm::Event& evt, const edm::EventSetup& es) override;

  double getDistInCM(double eta1, double phi1, double eta2, double phi2);
  std::pair<double, double> GetEtaPhiAtEcal(const edm::EventSetup& iSetup, double etaIP, double phiIP, double pT, int charge, double vtxZ);

 private:
	
  std::vector<edm::InputTag> pixelTracksSources_;
  edm::ParameterSet parameters;

  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tok_hlt_;
  edm::EDGetTokenT<l1extra::L1JetParticleCollection> tok_l1_;
  edm::EDGetTokenT<reco::VertexCollection> tok_vert_;

  std::vector<edm::EDGetTokenT<reco::TrackRef> > toks_pix_;

  double prelimCone_;
  double pixelIsolationConeSizeAtEC_;
  double vtxCutSeed_;
  double vtxCutIsol_;
  double tauAssocCone_;
  double tauUnbiasCone_;
  std::string bfield_;
  double minPTrackValue_;
  double maxPForIsolationValue_;
  double rEB_;
  double zEE_;
  double ebEtaBoundary_;
};


#endif
