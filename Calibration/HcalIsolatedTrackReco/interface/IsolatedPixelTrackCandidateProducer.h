#ifndef Calibration_IsolatedPixelTrackCandidateProducer_h
#define Calibration_IsolatedPixelTrackCandidateProducer_h

/* \class IsolatedPixelTrackCandidateProducer
 *
 *  
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"

//#include "DataFormats/Common/interface/Provenance.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"
#include "DataFormats/Common/interface/TriggerResults.h"
// L1Extra
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
///
//vertices
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutSetupFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapRecord.h"
//#include "DataFormats/L1GlobalTrigger/interface/L1GtLogicParser.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMapFwd.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerObjectMap.h"

class IsolatedPixelTrackCandidateProducer : public edm::stream::EDProducer<> {

public:

  IsolatedPixelTrackCandidateProducer (const edm::ParameterSet& ps);
  ~IsolatedPixelTrackCandidateProducer();
  

  virtual void beginRun(const edm::Run&, const edm::EventSetup&) override;
  virtual void produce(edm::Event& evt, const edm::EventSetup& es) override;
  
  double getDistInCM(double eta1, double phi1, double eta2, double phi2);
  std::pair<double, double> GetEtaPhiAtEcal(double etaIP, double phiIP, double pT, int charge, double vtxZ);

private:
  struct seedAtEC {
    seedAtEC(unsigned int i, bool f, double et, double fi) : index(i), ok(f), eta(et), phi(fi) { }
    unsigned int index;
    bool         ok;
    double       eta, phi;
  };

  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tok_hlt_;
  const edm::EDGetTokenT<l1extra::L1JetParticleCollection> tok_l1_;
  const edm::EDGetTokenT<reco::VertexCollection> tok_vert_;
  const std::vector<edm::EDGetTokenT<reco::TrackCollection> > toks_pix_;

  const std::string bfield_;
  const double prelimCone_;
  const double pixelIsolationConeSizeAtEC_;
  const double vtxCutSeed_;
  const double vtxCutIsol_;
  const double tauAssocCone_;
  const double tauUnbiasCone_;
  const double minPTrackValue_;
  const double maxPForIsolationValue_;
  const double ebEtaBoundary_;

  // these are read from the EventSetup, cannot be const
  double rEB_;
  double zEE_;
  double bfVal_;
};


#endif
