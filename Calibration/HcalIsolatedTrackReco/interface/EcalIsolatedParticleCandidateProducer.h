#ifndef Calibration_EcalIsolatedParticleCandidateProducer_h
#define Calibration_EcalIsolatedParticleCandidateProducer_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"
#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidate.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
//
// class decleration
//

class EcalIsolatedParticleCandidateProducer : public edm::global::EDProducer<> {
public:
  explicit EcalIsolatedParticleCandidateProducer(const edm::ParameterSet&);
  ~EcalIsolatedParticleCandidateProducer() override;

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
  void beginJob() override;
  void endJob() override;

  double InConeSize_;
  double OutConeSize_;
  double hitCountEthr_;
  double hitEthr_;

  edm::EDGetTokenT<l1extra::L1JetParticleCollection> tok_l1tau_;
  edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tok_hlt_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EB_;
  edm::EDGetTokenT<EcalRecHitCollection> tok_EE_;

  edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;

  // ----------member data ---------------------------
};

#endif
