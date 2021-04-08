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

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

//
// class decleration
//

class IsolatedEcalPixelTrackCandidateProducer : public edm::global::EDProducer<> {
public:
  explicit IsolatedEcalPixelTrackCandidateProducer(const edm::ParameterSet&);
  ~IsolatedEcalPixelTrackCandidateProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  const edm::EDGetTokenT<EcalRecHitCollection> tok_ee;
  const edm::EDGetTokenT<EcalRecHitCollection> tok_eb;
  const edm::EDGetTokenT<trigger::TriggerFilterObjectWithRefs> tok_trigcand;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tok_geom_;
  const double coneSizeEta0_;
  const double coneSizeEta1_;
  const double hitCountEthrEB_;
  const double hitEthrEB_;
  const double fachitCountEE_;
  const double hitEthrEE0_;
  const double hitEthrEE1_;
  const double hitEthrEE2_;
  const double hitEthrEE3_;
};

#endif
