// -*- C++ -*-
//
// Package:    EcalTBGeometryBuilder
// Class:      EcalTBGeometryBuilder
// 
/**\class EcalTBGeometryBuilder EcalTBGeometryBuilder.h tmp/EcalTBGeometryBuilder/interface/EcalTBGeometryBuilder.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jeremiah Mans
//         Created:  Mon Oct  3 11:35:27 CDT 2005
// $Id: EcalTBGeometryBuilder.cc,v 1.3 2010/03/26 19:51:48 sunanda Exp $
//
//


// user include files
#include "Geometry/EcalTestBeam/plugins/EcalTBGeometryBuilder.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


//
// constructors and destructor
//
EcalTBGeometryBuilder::EcalTBGeometryBuilder(const edm::ParameterSet& iConfig)
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced(this);

   //now do what ever other initialization is needed
}


EcalTBGeometryBuilder::~EcalTBGeometryBuilder()
{ 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
EcalTBGeometryBuilder::ReturnType
EcalTBGeometryBuilder::produce(const IdealGeometryRecord& iRecord)
{
   edm::ESHandle<CaloSubdetectorGeometry> pG;

   std::auto_ptr<CaloGeometry> pCaloGeom(new CaloGeometry());

   // TODO: Look for ECAL parts
   try {
     iRecord.get("EcalBarrel", pG); 
     pCaloGeom->setSubdetGeometry(DetId::Ecal,EcalBarrel,pG.product());
   } catch (...) {
     edm::LogWarning("MissingInput") << "No Ecal Barrel Geometry found";     
   }
   try {
     iRecord.get("EcalLaserPnDiode", pG); 
     pCaloGeom->setSubdetGeometry(DetId::Ecal,EcalLaserPnDiode,pG.product());
   } catch (...) {
     edm::LogWarning("MissingInput") << "No Ecal TB Hodoscope Geometry found";     
   }

   return pCaloGeom;
}
