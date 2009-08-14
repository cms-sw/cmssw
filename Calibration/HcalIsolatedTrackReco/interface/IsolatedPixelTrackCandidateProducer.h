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

  double getDistInCM(double eta1, double phi1, double eta2, double phi2);
  std::pair<double, double> GetEtaPhiAtEcal(double etaIP, double phiIP, double pT, int charge, double vtxZ);

 private:
	
  edm::InputTag hltGTseedlabel_;
  edm::InputTag l1eTauJetsSource_;
  edm::InputTag pixelTracksSource_;
  edm::InputTag vertexLabel_;
  edm::ParameterSet parameters;

  double pixelIsolationConeSize_;
  double pixelIsolationConeSizeAtEC_;
  double vtxCutSeed_;
  double vtxCutIsol_;
  double tauAssocCone_;
  double tauUnbiasCone_;
  double minPTrackValue_;
  double maxPForIsolationValue_;

};


#endif
