
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/EcalBarrelAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalEndcapAlgo/interface/EcalEndcapGeometry.h"
#include "FastSimulation/CalorimeterProperties/interface/Calorimeter.h"

#include "FastSimulation/CalorimeterProperties/interface/PreshowerLayer1Properties.h"
#include "FastSimulation/CalorimeterProperties/interface/PreshowerLayer2Properties.h"
#include "FastSimulation/CalorimeterProperties/interface/ECALBarrelProperties.h"
#include "FastSimulation/CalorimeterProperties/interface/ECALEndcapProperties.h"
#include "FastSimulation/CalorimeterProperties/interface/HCALBarrelProperties.h"
#include "FastSimulation/CalorimeterProperties/interface/HCALEndcapProperties.h"
#include "FastSimulation/CalorimeterProperties/interface/HCALForwardProperties.h"
#include "Geometry/HcalTowerAlgo/interface/HcalHardcodeGeometryLoader.h"



#include <iostream>

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
  myPreshowerLayer1Properties_  = new PreshowerLayer1Properties(fastDet); 
  myPreshowerLayer2Properties_  = new PreshowerLayer2Properties(fastDet);
  myECALBarrelProperties_       = new ECALBarrelProperties     (fastDet);
  myECALEndcapProperties_       = new ECALEndcapProperties     (fastDet);
  myHCALBarrelProperties_       = new HCALBarrelProperties     (fastDet);
  myHCALEndcapProperties_       = new HCALEndcapProperties     (fastDet);
  myHCALForwardProperties_      = new HCALForwardProperties    (fastDet);

  psLayer1Z_ = 303;
  psLayer2Z_ = 307;
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
	std::cout << " Calorimeter::hcalProperties : set myHCALForwardProperties" << std::endl;
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

void Calorimeter::setupGeometry(const edm::ESHandle<CaloGeometry>& pG)
{
  std::cout << " setupGeometry " << std::endl;
  EcalBarrelGeometry_ = dynamic_cast<const EcalBarrelGeometry*>(pG->getSubdetectorGeometry(DetId::Ecal,EcalBarrel));
  EcalEndcapGeometry_ = dynamic_cast<const EcalEndcapGeometry*>(pG->getSubdetectorGeometry(DetId::Ecal,EcalEndcap));
  HcalGeometry_ = pG->getSubdetectorGeometry(DetId::Hcal,HcalBarrel);

  // Takes a lot of time
  //  PreshowerGeometry_  = pG->getSubdetectorGeometry(DetId::Ecal,EcalPreshower);
}


DetId Calorimeter::getClosestCell(const HepPoint3D& point, bool ecal, bool central) const
{
  DetId result;
  //  std::cout << " In getClosestCell " << ecal << " " << central << std::endl;
  if(ecal)
    {
      if(central)
	{
	  //	  std::cout << "EcalBarrelGeometry_" << " " << EcalBarrelGeometry_ << std::endl;
	  result = EcalBarrelGeometry_->getClosestCell(GlobalPoint(point.x(),point.y(),point.z()));
	}
      else
	{
	  result = EcalEndcapGeometry_->getClosestCell(GlobalPoint(point.x(),point.y(),point.z()));
	}
    }
  else
    {
      result=HcalGeometry_->getClosestCell(GlobalPoint(point.x(),point.y(),point.z()));
    }
  
  //  std::cout << " done " << result.rawId() << std::endl;
  return result;
}
