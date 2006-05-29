// -*- C++ -*-
//
// Package:    RPCTrigger
// Class:      RPCTrigger
// 
/**\class RPCTrigger RPCTrigger.cc L1Trigger/RPCTrigger/src/RPCTrigger.cc

 Description: emulates rpc trigger

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Frueboes
//         Created:  Thu May 25 10:36:17 CEST 2006
// $Id$
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <FWCore/Framework/interface/ESHandle.h> // Handle to read geometry

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"


#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"

// L1RpcTrigger specific includes
#include "L1Trigger/RPCTrigger/interface/RPCTrigger.h"
#include "L1Trigger/RPCTrigger/src/RPCTriggerGeo.h"

RPCTrigger::RPCTrigger(const edm::ParameterSet& iConfig)
{
   // The data formats are not ready yet (V 2006), so we `produce` a fake data
   // to be able to run
  produces<int>("FakeTemp");

}


RPCTrigger::~RPCTrigger(){ }



void
RPCTrigger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
   // Build the trigger linksystem geometry;
  if (!linksystem.isGeometryBuild()){
    
    
    edm::ESHandle<RPCGeometry> rpcGeom;
    iSetup.get<MuonGeometryRecord>().get( rpcGeom );     
    linksystem.buildGeometry(rpcGeom);
  
  
  } 
 
   
   
   // Get RpcDigis, store them locally
   edm::Handle<RPCDigiCollection> rpcDigis;
   iEvent.getByType(rpcDigis);

   

   
   
   

}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCTrigger)
