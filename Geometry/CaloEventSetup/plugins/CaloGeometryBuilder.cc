// -*- C++ -*-
//
// Package:    CaloGeometryBuilder
// Class:      CaloGeometryBuilder
// 
/**\class CaloGeometryBuilder CaloGeometryBuilder.h tmp/CaloGeometryBuilder/interface/CaloGeometryBuilder.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Mon Oct  3 11:35:27 CDT 2005
// $Id: CaloGeometryBuilder.cc,v 1.2 2007/04/17 16:15:28 case Exp $
//
//


// user include files
#include "Geometry/CaloEventSetup/plugins/CaloGeometryBuilder.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


//
// constructors and destructor
//
CaloGeometryBuilder::CaloGeometryBuilder(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
}


CaloGeometryBuilder::~CaloGeometryBuilder()
{ 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
CaloGeometryBuilder::ReturnType
CaloGeometryBuilder::produce(const IdealGeometryRecord& iRecord)
{
   using namespace edm::es;
   edm::ESHandle<CaloSubdetectorGeometry> pG;

   std::auto_ptr<CaloGeometry> pCaloGeom(new CaloGeometry());

   // look for HCAL parts

   // assume 'HCAL' for all of HCAL.  
   // TODO: Eventually change to looking for "HO" and "HF" separately and fallback to HCAL
   try {
     iRecord.get("HCAL", pG); 
     pCaloGeom->setSubdetGeometry(DetId::Hcal,HcalBarrel,pG.product());
     pCaloGeom->setSubdetGeometry(DetId::Hcal,HcalEndcap,pG.product());
     pCaloGeom->setSubdetGeometry(DetId::Hcal,HcalOuter,pG.product());
     pCaloGeom->setSubdetGeometry(DetId::Hcal,HcalForward,pG.product());
   } catch (...) {
     edm::LogWarning("MissingInput") << "No HCAL Geometry found";
   }
   // look for zdc parts
   try {
     iRecord.get("ZDC", pG);
     pCaloGeom->setSubdetGeometry(DetId::Calo, HcalZDCDetId::SubdetectorId,pG.product());
   } catch(...) {
     edm::LogWarning("MissingInput") << "No ZDC Geometry found";
   }
     
   // TODO: Look for ECAL parts
   try {
     iRecord.get("EcalBarrel", pG); 
     pCaloGeom->setSubdetGeometry(DetId::Ecal,EcalBarrel,pG.product());
   } catch (...) {
     edm::LogWarning("MissingInput") << "No Ecal Barrel Geometry found";     
   }
   try {
     iRecord.get("EcalEndcap", pG); 
     pCaloGeom->setSubdetGeometry(DetId::Ecal,EcalEndcap,pG.product());
   } catch (...) {
     edm::LogWarning("MissingInput") << "No Ecal Endcap Geometry found";     
   }
   try {
     iRecord.get("EcalPreshower", pG); 
     pCaloGeom->setSubdetGeometry(DetId::Ecal,EcalPreshower,pG.product());
   } catch (...) {
     edm::LogWarning("MissingInput") << "No Ecal Preshower Geometry found";     
   }

   // look for TOWER parts
   try {
     iRecord.get("TOWER",pG);
     pCaloGeom->setSubdetGeometry(DetId::Calo,1,pG.product());
   } catch (...) {
     edm::LogWarning("MissingInput") << "No CaloTowers Geometry found";         
   }   

   return pCaloGeom;
}
