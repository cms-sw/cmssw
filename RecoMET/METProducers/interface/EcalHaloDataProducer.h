#ifndef RECOMET_METPRODUCERS_ECALHALODATAPRODUCER_H
#define RECOMET_METPRODUCERS_ECALHALODATAPRODUCER_H

/*
  [class]:  EcalHaloDataProducer
  [authors]: R. Remington, The University of Florida
  [description]: EDProducer which runs EcalHaloAlgo and store the EcalHaloData object to the event.
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

#include "DataFormats/METReco/interface/EcalHaloData.h"
#include "RecoMET/METAlgorithms/interface/EcalHaloAlgo.h"
//Included Classes (semi-alphabetical)
#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EcalDetId/interface/ESDetId.h"
#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitDefs.h"
#include "DataFormats/HepMCCandidate/interface/PdfInfo.h"
#include "DataFormats/GeometrySurface/interface/Cylinder.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"
#include "DataFormats/GeometrySurface/interface/Cone.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

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
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "TrackingTools/GeomPropagators/interface/AnalyticalPropagator.h"
#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

namespace reco {
  class EcalHaloDataProducer : public edm::stream::EDProducer<> {
  public:
    explicit EcalHaloDataProducer(const edm::ParameterSet&);
    ~EcalHaloDataProducer() override;

  private:
    void produce(edm::Event&, const edm::EventSetup&) override;

    //RecHit Level
    edm::InputTag IT_EBRecHit;
    edm::InputTag IT_EERecHit;
    edm::InputTag IT_ESRecHit;
    edm::InputTag IT_HBHERecHit;

    //Higher Level Reco
    edm::InputTag IT_SuperCluster;
    edm::InputTag IT_Photon;

    edm::EDGetTokenT<EBRecHitCollection> ebrechit_token_;
    edm::EDGetTokenT<EERecHitCollection> eerechit_token_;
    edm::EDGetTokenT<ESRecHitCollection> esrechit_token_;
    edm::EDGetTokenT<HBHERecHitCollection> hbherechit_token_;
    edm::EDGetTokenT<reco::SuperClusterCollection> supercluster_token_;
    edm::EDGetTokenT<reco::PhotonCollection> photon_token_;
    edm::ESGetToken<CaloGeometry, CaloGeometryRecord> calogeometry_token_;
    EcalHaloAlgo EcalAlgo;

    float EBRecHitEnergyThreshold;
    float EERecHitEnergyThreshold;
    float ESRecHitEnergyThreshold;
    float SumEcalEnergyThreshold;
    int NHitsEcalThreshold;

    double RoundnessCut;
    double AngleCut;
  };
}  // namespace reco

#endif
