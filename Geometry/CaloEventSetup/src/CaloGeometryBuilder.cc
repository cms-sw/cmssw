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
// $Id: CaloGeometryBuilder.cc,v 1.6 2006/02/05 12:08:30 meridian Exp $
//
//


// user include files
#include "Geometry/CaloEventSetup/src/CaloGeometryBuilder.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

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
   iRecord.get("HCAL", pG); 
   pCaloGeom->setSubdetGeometry(DetId::Hcal,HcalBarrel,pG.product());
   pCaloGeom->setSubdetGeometry(DetId::Hcal,HcalEndcap,pG.product());
   pCaloGeom->setSubdetGeometry(DetId::Hcal,HcalOuter,pG.product());
   pCaloGeom->setSubdetGeometry(DetId::Hcal,HcalForward,pG.product());
   
   // TODO: Look for ECAL parts
   iRecord.get("EcalBarrel", pG); 
   pCaloGeom->setSubdetGeometry(DetId::Ecal,EcalBarrel,pG.product());
   iRecord.get("EcalEndcap", pG); 
   pCaloGeom->setSubdetGeometry(DetId::Ecal,EcalEndcap,pG.product());
   iRecord.get("EcalPreshower", pG); 
   pCaloGeom->setSubdetGeometry(DetId::Ecal,EcalPreshower,pG.product());

   // look for TOWER parts
   iRecord.get("TOWER",pG);
   pCaloGeom->setSubdetGeometry(DetId::Calo,1,pG.product());
   
   return pCaloGeom;
}
