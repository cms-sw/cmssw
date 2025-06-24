#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "FastSimulation/CalorimeterProperties/interface/Calorimeter.h"
#include "FastSimulation/CalorimeterProperties/interface/PreshowerLayer1Properties.h"
#include "FastSimulation/CalorimeterProperties/interface/PreshowerLayer2Properties.h"
#include "FastSimulation/CalorimeterProperties/interface/ECALBarrelProperties.h"
#include "FastSimulation/CalorimeterProperties/interface/ECALEndcapProperties.h"
#include "FastSimulation/CalorimeterProperties/interface/HCALBarrelProperties.h"
#include "FastSimulation/CalorimeterProperties/interface/HCALEndcapProperties.h"
#include "FastSimulation/CalorimeterProperties/interface/HCALForwardProperties.h"
#include "Geometry/HcalTowerAlgo/interface/HcalHardcodeGeometryLoader.h"

Calorimeter::Calorimeter()
    : EcalBarrelGeometry_(nullptr),
      EcalEndcapGeometry_(nullptr),
      HcalGeometry_(nullptr),
      PreshowerGeometry_(nullptr) {
}

Calorimeter::Calorimeter(const edm::ParameterSet& fastCalo)
    : Calorimeter(fastCalo.getParameter<edm::ParameterSet>("CalorimeterProperties"),
                  fastCalo.getParameter<edm::ParameterSet>("ForwardCalorimeterProperties")) {
}

Calorimeter::Calorimeter(const edm::ParameterSet& fastDet, const edm::ParameterSet& fastDetHF)
    : myPreshowerLayer1Properties_(std::make_unique<PreshowerLayer1Properties>(fastDet)),
      myPreshowerLayer2Properties_(std::make_unique<PreshowerLayer2Properties>(fastDet)),
      myECALBarrelProperties_(std::make_unique<ECALBarrelProperties>(fastDet)),
      myECALEndcapProperties_(std::make_unique<ECALEndcapProperties>(fastDet)),
      myHCALBarrelProperties_(std::make_unique<HCALBarrelProperties>(fastDet)),
      myHCALEndcapProperties_(std::make_unique<HCALEndcapProperties>(fastDet)),
      myHCALForwardProperties_(std::make_unique<HCALForwardProperties>(fastDetHF)),
      EcalBarrelGeometry_(nullptr),
      EcalEndcapGeometry_(nullptr),
      HcalGeometry_(nullptr),
      PreshowerGeometry_(nullptr) {
}

Calorimeter::~Calorimeter() { }

const ECALProperties* Calorimeter::ecalProperties(int onEcal) const {
  if (onEcal) {
    if (onEcal == 1)
      return myECALBarrelProperties_.get();
    else
      return myECALEndcapProperties_.get();
  } else
    return nullptr;
}

const HCALProperties* Calorimeter::hcalProperties(int onHcal) const {
  if (onHcal) {
    if (onHcal == 1)
      return myHCALBarrelProperties_.get();
    else if (onHcal == 2)
      return myHCALEndcapProperties_.get();
    else {
      return myHCALForwardProperties_.get();
      edm::LogInfo("CalorimeterProperties")
          << " Calorimeter::hcalProperties : set myHCALForwardProperties" << std::endl;
    }
  } else
    return nullptr;
}

const PreshowerLayer1Properties* Calorimeter::layer1Properties(int onLayer1) const {
  if (onLayer1)
    return myPreshowerLayer1Properties_.get();
  else
    return nullptr;
}

const PreshowerLayer2Properties* Calorimeter::layer2Properties(int onLayer2) const {
  if (onLayer2)
    return myPreshowerLayer2Properties_.get();
  else
    return nullptr;
}

void Calorimeter::setupGeometry(const CaloGeometry& pG) {
  edm::LogInfo("CalorimeterProperties") << " setupGeometry " << std::endl;
  EcalBarrelGeometry_ = dynamic_cast<const EcalBarrelGeometry*>(pG.getSubdetectorGeometry(DetId::Ecal, EcalBarrel));
  EcalEndcapGeometry_ = dynamic_cast<const EcalEndcapGeometry*>(pG.getSubdetectorGeometry(DetId::Ecal, EcalEndcap));
  HcalGeometry_ = pG.getSubdetectorGeometry(DetId::Hcal, HcalBarrel);
  // Takes a lot of time
  PreshowerGeometry_ =
      dynamic_cast<const EcalPreshowerGeometry*>(pG.getSubdetectorGeometry(DetId::Ecal, EcalPreshower));
}

void Calorimeter::setupTopology(const CaloTopology& theTopology) {
  EcalBarrelTopology_ = theTopology.getSubdetectorTopology(DetId::Ecal, EcalBarrel);
  EcalEndcapTopology_ = theTopology.getSubdetectorTopology(DetId::Ecal, EcalEndcap);
}

const CaloSubdetectorGeometry* Calorimeter::getEcalGeometry(int subdetn) const {
  if (subdetn == 1)
    return EcalBarrelGeometry_;
  if (subdetn == 2)
    return EcalEndcapGeometry_;
  if (subdetn == 3)
    return PreshowerGeometry_;
  edm::LogWarning("Calorimeter") << "Requested an invalid ECAL subdetector geometry: " << subdetn << std::endl;
  return nullptr;
}

const CaloSubdetectorTopology* Calorimeter::getEcalTopology(int subdetn) const {
  if (subdetn == 1)
    return EcalBarrelTopology_;
  if (subdetn == 2)
    return EcalEndcapTopology_;
  edm::LogWarning("Calorimeter") << "Requested an invalid ECAL subdetector topology: " << subdetn << std::endl;
  return nullptr;
}
