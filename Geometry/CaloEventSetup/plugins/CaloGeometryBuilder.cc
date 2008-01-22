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
// $Id: CaloGeometryBuilder.cc,v 1.5 2007/11/15 17:13:26 fabiocos Exp $
//
//


// user include files
#include "Geometry/CaloEventSetup/plugins/CaloGeometryBuilder.h"
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
   
   theCaloList = iConfig.getParameter< std::vector<std::string> >("SelectedCalos");
   if ( theCaloList.size() == 0 ) throw cms::Exception("Configuration") << "No calorimeter specified for geometry, aborting";

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

   // loop on selected calorimeters

   for ( std::vector<std::string>::const_iterator ite = theCaloList.begin(); ite != theCaloList.end(); ite++ ) {

     // look for HCAL parts
     // assume 'HCAL' for all of HCAL.  
     if ( (*ite) == "HCAL" ) {  
       edm::LogInfo("CaloGeometryBuilder") << "Building HCAL reconstruction geometry";
       iRecord.get("HCAL", pG); 
       pCaloGeom->setSubdetGeometry(DetId::Hcal,HcalBarrel,pG.product());
       pCaloGeom->setSubdetGeometry(DetId::Hcal,HcalEndcap,pG.product());
       pCaloGeom->setSubdetGeometry(DetId::Hcal,HcalOuter,pG.product());
       pCaloGeom->setSubdetGeometry(DetId::Hcal,HcalForward,pG.product());
     } 
     
     // look for zdc parts
     else if ( (*ite) == "ZDC" ) {
       edm::LogInfo("CaloGeometryBuilder") << "Building ZDC reconstruction geometry";
       iRecord.get("ZDC", pG);
       pCaloGeom->setSubdetGeometry(DetId::Calo, HcalZDCDetId::SubdetectorId,pG.product());
     }
     
     // look for Ecal Barrel
     else if ( (*ite) == "EcalBarrel" ) {
       edm::LogInfo("CaloGeometryBuilder") << "Building EcalBarrel reconstruction geometry";
       iRecord.get("EcalBarrel", pG); 
       pCaloGeom->setSubdetGeometry(DetId::Ecal,EcalBarrel,pG.product());
     }

     // look for Ecal Endcap
     else if ( (*ite) == "EcalEndcap" ) {
       edm::LogInfo("CaloGeometryBuilder") << "Building EcalEndcap reconstruction geometry";
       iRecord.get("EcalEndcap", pG); 
       pCaloGeom->setSubdetGeometry(DetId::Ecal,EcalEndcap,pG.product());
     }

     // look for Ecal Preshower
     else if ( (*ite) == "EcalPreshower" ) {
       edm::LogInfo("CaloGeometryBuilder") << "Building EcalPreshower reconstruction geometry";
       iRecord.get("EcalPreshower", pG); 
       pCaloGeom->setSubdetGeometry(DetId::Ecal,EcalPreshower,pG.product());
     }

     // look for TOWER parts

     else if ( (*ite) == "TOWER" ) {
       edm::LogInfo("CaloGeometryBuilder") << "Building TOWER reconstruction geometry";
       iRecord.get("TOWER",pG);
       pCaloGeom->setSubdetGeometry(DetId::Calo,1,pG.product());
     }

     else { 
       edm::LogWarning("CaloGeometryBuilder") << "Reconstrcution geometry requested for a not implemented sub-detector: " << (*ite); 
     }

   }

   return pCaloGeom;
}
