#ifndef RECOMET_METPRODUCERS_HCALHALODATAPRODUCER_H
#define RECOMET_METPRODUCERS_HCALHALODATAPRODUCER_H

/*
  [class]:  HcalHaloDataProducer
  [authors]: R. Remington, The University of Florida
  [description]: EDProducer which runs HcalHaloAlgo and stores HcalHaloData object to the event. 
  [date]: October 15, 2009
*/

//Standard C++ classes
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <utility>
#include <ostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <memory>
#include <iomanip>
#include <cstdlib>

// user include files
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/METReco/interface/HcalHaloData.h"
#include "RecoMET/METAlgorithms/interface/HcalHaloAlgo.h"
//Included Classes (semi-alphabetical)
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cone.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

namespace reco {
  class HcalHaloDataProducer : public edm::stream::EDProducer<> {
  public:
    explicit HcalHaloDataProducer(const edm::ParameterSet&);
    ~HcalHaloDataProducer() override;

  private:
    void produce(edm::Event&, const edm::EventSetup&) override;

    //RecHit Level
    edm::InputTag IT_HBHERecHit;
    edm::InputTag IT_HORecHit;
    edm::InputTag IT_HFRecHit;
    edm::InputTag IT_CaloTowers;
    edm::InputTag IT_EBRecHit;
    edm::InputTag IT_EERecHit;

    edm::EDGetTokenT<EBRecHitCollection> ebrechit_token_;
    edm::EDGetTokenT<EERecHitCollection> eerechit_token_;
    edm::EDGetTokenT<HBHERecHitCollection> hbherechit_token_;
    edm::EDGetTokenT<HFRecHitCollection> hfrechit_token_;
    edm::EDGetTokenT<CaloTowerCollection> calotower_token_;
    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> calogeometry_token_;
    HcalHaloAlgo HcalAlgo;

    float HBRecHitEnergyThreshold;
    float HERecHitEnergyThreshold;
    float SumHcalEnergyThreshold;
    int NHitsHcalThreshold;
  };
}  // namespace reco

#endif
