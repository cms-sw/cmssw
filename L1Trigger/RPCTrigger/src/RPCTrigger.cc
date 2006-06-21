/** \file RPCTrigger.cc
 *
 *  $Date: 2006/06/19 15:28:49 $
 *  $Revision: 1.8 $
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
#include "FWCore/MessageLogger/interface/MessageLogger.h"

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
  
  
  std::string patternsDirName = iConfig.getParameter<std::string>("RPCPatternsDir");
  
  //m_pacManager.Init("/afs/cern.ch/user/f/fruboes/public/patterns/", _12_PACS_PER_TOWER);
    
  m_pacManager.Init(patternsDirName, _12_PACS_PER_TOWER);
  
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

    edm::LogInfo("RPC") << "Building RPC links map for a RPCTrigger";
    edm::ESHandle<RPCGeometry> rpcGeom;
    iSetup.get<MuonGeometryRecord>().get( rpcGeom );     
    theLinksystem.buildGeometry(rpcGeom);
    edm::LogInfo("RPC") << "RPC links map for a RPCTrigger built";

  } 

  
  edm::Handle<RPCDigiCollection> rpcDigis;
  iEvent.getByType(rpcDigis);
  
  L1RpcLogConesVec ActiveCones = theLinksystem.getCones(rpcDigis);
  
  L1RpcTBMuonsVec2 finalMuons = m_pacTrigger->RunEvent(ActiveCones);

  // Fill out the products
  std::vector<L1MuRegionalCand> RPCb;
  std::vector<L1MuRegionalCand> RPCf;
  
  // TODO: check which one is barell
  for(unsigned int iMu = 0; iMu < finalMuons[0].size(); iMu++)
  {
    L1MuRegionalCand l1Cand;
    
    //RPCParam::L1RpcConeCrdnts cone = finalMuons[0][iMu].GetConeCrdnts();
    
    l1Cand.setQualityPacked(finalMuons[0][iMu].GetQuality());
    l1Cand.setPtPacked(finalMuons[0][iMu].GetPtCode());
    
    RPCb.push_back(l1Cand);
  }

  for(unsigned int iMu = 0; iMu < finalMuons[1].size(); iMu++)
  {
    L1MuRegionalCand l1Cand;
    
    //RPCParam::L1RpcConeCrdnts cone = finalMuons[0][iMu].GetConeCrdnts();
    
    l1Cand.setQualityPacked(finalMuons[0][iMu].GetQuality());
    l1Cand.setPtPacked(finalMuons[0][iMu].GetPtCode());
    
    RPCf.push_back(l1Cand);
  }

  std::auto_ptr<std::vector<L1MuRegionalCand> > candBarell(new std::vector<L1MuRegionalCand>);
  candBarell->insert(candBarell->end(), RPCb.begin(), RPCb.end());
  
  std::auto_ptr<std::vector<L1MuRegionalCand> > candForward(new std::vector<L1MuRegionalCand>);
  candForward->insert(candForward->end(), RPCf.begin(), RPCf.end());
  
  iEvent.put(candBarell, "RPCb");
  iEvent.put(candForward, "RPCf");
  
  LogDebug("RPCTrigger") << "-----------------------------" << std::endl;
  for(unsigned int iMu = 0; iMu < finalMuons[0].size(); iMu++)
  {
    LogDebug("RPCTrigger") << "Found muonf of pt " << finalMuons[0][iMu].GetPtCode() << std::endl;
  }

  for(unsigned int iMu = 0; iMu < finalMuons[1].size(); iMu++)
  {
    LogDebug("RPCTrigger") << "Found muonf of pt " << finalMuons[1][iMu].GetPtCode() << std::endl;
  }


}

//define this as a plug-in
DEFINE_FWK_MODULE(RPCTrigger)
