/** \file RPCTrigger.cc
 *
 *  $Date: 2006/06/16 11:07:14 $
 *  $Revision: 1.6 $
 *  \author Tomasz Fruboes
 */


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
#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuRegionalCand.h"


RPCTrigger::RPCTrigger(const edm::ParameterSet& iConfig)
{
  produces<std::vector<L1MuRegionalCand> >("RPCb");
  produces<std::vector<L1MuRegionalCand> >("RPCf");
  
  m_pacManager.Init("/afs/cern.ch/user/f/fruboes/public/patterns/", _12_PACS_PER_TOWER);
    
  m_trigConfig = new L1RpcBasicTrigConfig(&m_pacManager);
  
  m_trigConfig->SetDebugLevel(0);
  
  m_pacTrigger = new L1RpcPacTrigger(m_trigConfig);
  
}


RPCTrigger::~RPCTrigger(){ 
  delete m_pacTrigger;
  delete m_trigConfig;
}



void
RPCTrigger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  // Build the trigger linksystem geometry;
  if (!theLinksystem.isGeometryBuilt()){

    edm::ESHandle<RPCGeometry> rpcGeom;
    iSetup.get<MuonGeometryRecord>().get( rpcGeom );     
    theLinksystem.buildGeometry(rpcGeom);

  } 

  
  edm::Handle<RPCDigiCollection> rpcDigis;
  iEvent.getByType(rpcDigis);
  
  L1RpcLogConesVec ActiveCones = theLinksystem.getCones(rpcDigis);
  
  L1RpcTBMuonsVec2 finalMuons = m_pacTrigger->RunEvent(ActiveCones);

  std::cout << "-----------------------------" << std::endl;

  for(unsigned int iMu = 0; iMu < finalMuons[0].size(); iMu++)
  {
    std::cout << "Found muonf of pt " << finalMuons[0][iMu].GetPtCode() << std::endl;
  }

  for(unsigned int iMu = 0; iMu < finalMuons[1].size(); iMu++)
  {
    std::cout << "Found muonf of pt " << finalMuons[1][iMu].GetPtCode() << std::endl;
  }



}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCTrigger)
