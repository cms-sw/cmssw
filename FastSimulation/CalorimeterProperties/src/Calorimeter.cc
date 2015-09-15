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

Calorimeter::Calorimeter():
  myPreshowerLayer1Properties_(NULL),
  myPreshowerLayer2Properties_(NULL),  
  myECALBarrelProperties_     (NULL),  
  myECALEndcapProperties_     (NULL),  
  myHCALBarrelProperties_     (NULL),  
  myHCALEndcapProperties_     (NULL),  
  myHCALForwardProperties_    (NULL),
  EcalBarrelGeometry_         (NULL),
  EcalEndcapGeometry_         (NULL),
  HcalGeometry_               (NULL),
  PreshowerGeometry_          (NULL)
{
;
}

Calorimeter::Calorimeter(const edm::ParameterSet& fastCalo):
  myPreshowerLayer1Properties_(NULL),
  myPreshowerLayer2Properties_(NULL),  
  myECALBarrelProperties_     (NULL),  
  myECALEndcapProperties_     (NULL),  
  myHCALBarrelProperties_     (NULL),  
  myHCALEndcapProperties_     (NULL),  
  myHCALForwardProperties_    (NULL),
  EcalBarrelGeometry_        (NULL),
  EcalEndcapGeometry_          (NULL),
  HcalGeometry_               (NULL),
  PreshowerGeometry_          (NULL)  
{
  edm::ParameterSet fastDet = fastCalo.getParameter<edm::ParameterSet>("CalorimeterProperties");
  edm::ParameterSet fastDetHF = fastCalo.getParameter<edm::ParameterSet>("ForwardCalorimeterProperties");

  myPreshowerLayer1Properties_  = new PreshowerLayer1Properties(fastDet); 
  myPreshowerLayer2Properties_  = new PreshowerLayer2Properties(fastDet);
  myECALBarrelProperties_       = new ECALBarrelProperties     (fastDet);
  myECALEndcapProperties_       = new ECALEndcapProperties     (fastDet);
  myHCALBarrelProperties_       = new HCALBarrelProperties     (fastDet);
  myHCALEndcapProperties_       = new HCALEndcapProperties     (fastDet);
  myHCALForwardProperties_      = new HCALForwardProperties    (fastDetHF);

}

Calorimeter::~Calorimeter()
{
  if(myPreshowerLayer1Properties_        )  delete myPreshowerLayer1Properties_      ;
  if(myPreshowerLayer2Properties_        )  delete myPreshowerLayer2Properties_      ;
  if(myECALBarrelProperties_             )  delete myECALBarrelProperties_           ;
  if(myECALEndcapProperties_             )  delete myECALEndcapProperties_           ;
  if(myHCALBarrelProperties_             )  delete myHCALBarrelProperties_           ;
  if(myHCALEndcapProperties_             )  delete myHCALEndcapProperties_           ;
  if(myHCALForwardProperties_            )  delete myHCALForwardProperties_          ;
}

const ECALProperties*
Calorimeter::ecalProperties(int onEcal) const {
  if ( onEcal ) {
    if ( onEcal == 1 ) 
      return myECALBarrelProperties_;
    else
      return myECALEndcapProperties_;
  } else
    return NULL;
}

const HCALProperties*
Calorimeter::hcalProperties(int onHcal) const {
  if ( onHcal ) {
    if ( onHcal == 1 ) 
      return myHCALBarrelProperties_;
    else 
      if ( onHcal == 2 ) 
	return myHCALEndcapProperties_;
      else {
	return myHCALForwardProperties_;
	edm::LogInfo("CalorimeterProperties") << " Calorimeter::hcalProperties : set myHCALForwardProperties" << std::endl;
      }
  } else
    return NULL;
}

const PreshowerLayer1Properties*
Calorimeter::layer1Properties(int onLayer1) const {
  if ( onLayer1 ) 
    return myPreshowerLayer1Properties_;
  else
    return NULL;
}

const PreshowerLayer2Properties*
Calorimeter::layer2Properties(int onLayer2) const {
  if ( onLayer2 ) 
    return myPreshowerLayer2Properties_;
  else
    return NULL;
}

void Calorimeter::setupGeometry(const CaloGeometry& pG)
{
  edm::LogInfo("CalorimeterProperties") << " setupGeometry " << std::endl;
  EcalBarrelGeometry_ = dynamic_cast<const EcalBarrelGeometry*>(pG.getSubdetectorGeometry(DetId::Ecal,EcalBarrel));
  EcalEndcapGeometry_ = dynamic_cast<const EcalEndcapGeometry*>(pG.getSubdetectorGeometry(DetId::Ecal,EcalEndcap));
  HcalGeometry_ = pG.getSubdetectorGeometry(DetId::Hcal,HcalBarrel);
  // Takes a lot of time
  PreshowerGeometry_  = dynamic_cast<const EcalPreshowerGeometry*>(pG.getSubdetectorGeometry(DetId::Ecal,EcalPreshower));
}

void Calorimeter::setupTopology(const CaloTopology& theTopology)
{
  EcalBarrelTopology_ = theTopology.getSubdetectorTopology(DetId::Ecal,EcalBarrel);
  EcalEndcapTopology_ = theTopology.getSubdetectorTopology(DetId::Ecal,EcalEndcap);
}



const CaloSubdetectorGeometry * Calorimeter::getEcalGeometry(int subdetn) const
{
  if(subdetn==1) return EcalBarrelGeometry_;
  if(subdetn==2) return EcalEndcapGeometry_;
  if(subdetn==3) return PreshowerGeometry_;
  edm::LogWarning("Calorimeter") << "Requested an invalid ECAL subdetector geometry: " << subdetn << std::endl;
  return 0;
}

const CaloSubdetectorTopology * Calorimeter::getEcalTopology(int subdetn) const
{
  if(subdetn==1) return EcalBarrelTopology_;
  if(subdetn==2) return EcalEndcapTopology_;
  edm::LogWarning("Calorimeter") << "Requested an invalid ECAL subdetector topology: " << subdetn << std::endl;
  return 0;
}
