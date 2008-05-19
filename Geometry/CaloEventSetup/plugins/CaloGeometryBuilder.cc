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
// $Id: CaloGeometryBuilder.cc,v 1.7 2008/04/21 22:14:19 heltsley Exp $
//
//


// user include files
#include "Geometry/CaloEventSetup/plugins/CaloGeometryBuilder.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//
// member functions
//
CaloGeometryBuilder::CaloGeometryBuilder( const edm::ParameterSet& iConfig )
{
   //the following line is needed to tell the framework what
   // data is being produced
   setWhatProduced( this, 
		    &CaloGeometryBuilder::produceAligned );

// disable
//   setWhatProduced( this, 
//		    &CaloGeometryBuilder::produceIdeal ) ;

   //now do what ever other initialization is needed
   
   theCaloList = iConfig.getParameter< std::vector<std::string> >("SelectedCalos");
   if ( theCaloList.size() == 0 ) throw cms::Exception("Configuration") 
      << "No calorimeter specified for geometry, aborting";
}

// ------------ method called to produce the data  ------------

CaloGeometryBuilder::ReturnType
CaloGeometryBuilder::produceIdeal( const IdealGeometryRecord& iRecord )
{
   edm::ESHandle<CaloSubdetectorGeometry> pG;

   ReturnType pCalo ( new CaloGeometry() );

   // loop on selected calorimeters

   for ( std::vector<std::string>::const_iterator ite ( theCaloList.begin() );
	 ite != theCaloList.end(); ite++ ) 
   {
      // look for HCAL parts
      // assume 'HCAL' for all of HCAL.  
      if ( (*ite) == "HCAL" ) 
      {  
	 edm::LogInfo("CaloGeometryBuilder") << "Building IDEAL HCAL reconstruction geometry";
	 iRecord.get("HCAL", pG); 
	 pCalo->setSubdetGeometry(DetId::Hcal,HcalBarrel,pG.product());
	 pCalo->setSubdetGeometry(DetId::Hcal,HcalEndcap,pG.product());
	 pCalo->setSubdetGeometry(DetId::Hcal,HcalOuter,pG.product());
	 pCalo->setSubdetGeometry(DetId::Hcal,HcalForward,pG.product());
      } 
     
      // look for zdc parts
      else if ( (*ite) == "ZDC" ) 
      {
	 edm::LogInfo("CaloGeometryBuilder") << "Building IDEAL ZDC reconstruction geometry";
	 iRecord.get("ZDC", pG);
	 pCalo->setSubdetGeometry(DetId::Calo, HcalZDCDetId::SubdetectorId,pG.product());
      }
     
      // look for Ecal Barrel
      else if ( (*ite) == "EcalBarrel" ) 
      {
	 edm::LogInfo("CaloGeometryBuilder") << "Building IDEAL EcalBarrel reconstruction geometry";
	 iRecord.get("EcalBarrel", pG); 
	 pCalo->setSubdetGeometry(DetId::Ecal,EcalBarrel,pG.product());
      }

      // look for Ecal Endcap
      else if ( (*ite) == "EcalEndcap" ) 
      {
	 edm::LogInfo("CaloGeometryBuilder") << "Building IDEAL EcalEndcap reconstruction geometry";
	 iRecord.get("EcalEndcap", pG); 
	 pCalo->setSubdetGeometry(DetId::Ecal,EcalEndcap,pG.product());
      }

      // look for Ecal Preshower
      else if ( (*ite) == "EcalPreshower" ) 
      {
	 edm::LogInfo("CaloGeometryBuilder") << "Building IDEAL EcalPreshower reconstruction geometry";
	 iRecord.get("EcalPreshower", pG); 
	 pCalo->setSubdetGeometry(DetId::Ecal,EcalPreshower,pG.product());
      }
      else if ( (*ite) == "TOWER" ) 
      {
	 edm::LogInfo("CaloGeometryBuilder") << "Building TOWER reconstruction geometry";
	 iRecord.get("TOWER",pG);
	 pCalo->setSubdetGeometry(DetId::Calo,1,pG.product());
      }
      else 
      { 
	 edm::LogWarning("CaloGeometryBuilder") 
	    << "IDEAL Reconstrcution geometry requested for a not implemented sub-detector: " 
	    << (*ite); 
      }
   }
   return pCalo;
}

CaloGeometryBuilder::ReturnType
CaloGeometryBuilder::produceAligned( const CaloGeometryRecord& iRecord )
{
   edm::ESHandle< CaloSubdetectorGeometry > pG;

   ReturnType pCalo ( new CaloGeometry() ) ;

   // loop on selected calorimeters

   for ( std::vector<std::string>::const_iterator ite ( theCaloList.begin() ) ;
	 ite != theCaloList.end(); ++ite ) 
   {
      // look for HCAL parts
      // assume 'HCAL' for all of HCAL.  
      if ( (*ite) == "HCAL" ) 
      {
	 edm::LogInfo("CaloGeometryBuilder") << "Building HCAL reconstruction geometry";

	 iRecord.getRecord< HcalGeometryRecord >().get("HCAL", pG);  
	 pCalo->setSubdetGeometry( DetId::Hcal, HcalBarrel , pG.product() );
	 pCalo->setSubdetGeometry( DetId::Hcal, HcalEndcap , pG.product() );
	 pCalo->setSubdetGeometry( DetId::Hcal, HcalOuter  , pG.product() );
	 pCalo->setSubdetGeometry( DetId::Hcal, HcalForward, pG.product() );
      }
      // look for zdc parts
      else if ( (*ite) == "ZDC" ) 
      {
	 edm::LogInfo("CaloGeometryBuilder") << "Building ZDC reconstruction geometry";
	 iRecord.getRecord< ZDCGeometryRecord >().get("ZDC", pG); 
	 pCalo->setSubdetGeometry( DetId::Calo, HcalZDCDetId::SubdetectorId,pG.product());
      }
      // look for Ecal Barrel
      else if ( (*ite) == "EcalBarrel" ) 
      {
	 edm::LogInfo("CaloGeometryBuilder") << "Building EcalBarrel reconstruction geometry";
	 iRecord.getRecord<EcalBarrelGeometryRecord>().get("EcalBarrel", pG); 
	 pCalo->setSubdetGeometry(DetId::Ecal,EcalBarrel,pG.product());
      }
      // look for Ecal Endcap
      else if ( (*ite) == "EcalEndcap" ) 
      {
	 edm::LogInfo("CaloGeometryBuilder") << "Building EcalEndcap reconstruction geometry";
	 iRecord.getRecord<EcalEndcapGeometryRecord>().get("EcalEndcap", pG); 
	 pCalo->setSubdetGeometry(DetId::Ecal,EcalEndcap,pG.product());
      }
      // look for Ecal Preshower
      else if ( (*ite) == "EcalPreshower" ) 
      {
	 edm::LogInfo("CaloGeometryBuilder") << "Building EcalPreshower reconstruction geometry";
	 iRecord.getRecord<EcalPreshowerGeometryRecord>().get("EcalPreshower", pG); 
	 pCalo->setSubdetGeometry(DetId::Ecal,EcalPreshower,pG.product());
      }
      // look for TOWER parts
      else if ( (*ite) == "TOWER" ) 
      {
	 edm::LogInfo("CaloGeometryBuilder") << "Building TOWER reconstruction geometry";
	 iRecord.getRecord<IdealGeometryRecord>().get("TOWER",pG);
	 pCalo->setSubdetGeometry(DetId::Calo,1,pG.product());
      }
      else 
      {
	 edm::LogWarning("CaloGeometryBuilder") 
	    << "Reconstrcution geometry requested for a not implemented sub-detector: " 
	    << (*ite); 
      }
   }
   return pCalo ;
}
