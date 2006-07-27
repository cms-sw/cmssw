/** \file RPCTrigger.cc
 *
 *  $Date: 2006/07/25 12:44:00 $
 *  $Revision: 1.13 $
 *  \author Tomasz Fruboes
 */
#include "L1Trigger/RPCTrigger/interface/RPCTrigger.h"

//#define ML_DEBUG 





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

  int maxFiredPlanes = 0;
  
  for (unsigned int i=0;i<ActiveCones.size();i++){
      int fpCnt = ActiveCones[i].GetFiredPlanesCnt();
      if (fpCnt > maxFiredPlanes)
         maxFiredPlanes = fpCnt;
  }

  // Fill out the products
  // finalMuons[0]=barell, finalMuons[1]=endcap
  LogDebug("RPCTrigger") << "---Filling candindates in new event--- " 
                         << maxFiredPlanes << std::endl;
  
  std::vector<L1MuRegionalCand> RPCb = giveFinallCandindates(finalMuons[0],1);
  std::vector<L1MuRegionalCand> RPCf = giveFinallCandindates(finalMuons[1],3);;
    
  std::auto_ptr<std::vector<L1MuRegionalCand> > candBarell(new std::vector<L1MuRegionalCand>);
  candBarell->insert(candBarell->end(), RPCb.begin(), RPCb.end());
  
  std::auto_ptr<std::vector<L1MuRegionalCand> > candForward(new std::vector<L1MuRegionalCand>);
  candForward->insert(candForward->end(), RPCf.begin(), RPCf.end());
  
  iEvent.put(candBarell, "RPCb");
  iEvent.put(candForward, "RPCf");
  
}
///////////////////////////////////////////////////////////////////////////////
/**
 *
 * \brief Returns vector of L1MuRegionalCand (input of L1GMT)
 * \note - type is defined in L1MuRegionalCand 1 - barell, 3 - forward
 * \todo - we use offset value of 5 deegres. It should be stored centrally.
 *
 */
///////////////////////////////////////////////////////////////////////////////
std::vector<L1MuRegionalCand> RPCTrigger::giveFinallCandindates(L1RpcTBMuonsVec finalMuons, short type){

  std::vector<L1MuRegionalCand> RPCCand;
  
  for(unsigned int iMu = 0; iMu < finalMuons.size(); iMu++)
  {
    L1MuRegionalCand l1Cand;
    
    
    l1Cand.setQualityPacked(finalMuons[iMu].GetQuality());
    l1Cand.setPtPacked(finalMuons[iMu].GetPtCode());
    
    l1Cand.setType(type); 
    
    int charge=finalMuons[iMu].GetSign();
    
    if (charge == 0)  // negative
      l1Cand.setChargePacked(1);
    else  
      l1Cand.setChargePacked(0);
    
    rpcparam::L1RpcConeCrdnts cone = finalMuons[iMu].GetConeCrdnts();    
    
    int pac = cone.LogSector*12+cone.LogSegment;
    const float pi = 3.14159265;
    const float offset = 5*(2*pi/360); // redefinition! Defined also in RPCRingFromRolls::phiMapCompare
    float phi = 2*pi*pac/144-offset;
    if (phi<0)
      phi+=2*pi;
    
    l1Cand.setPhiValue(phi);
    
    float eta = L1RpcConst::etaFromTowerNum(cone.Tower);
    
    l1Cand.setEtaValue(eta);
    
    RPCCand.push_back(l1Cand);
        
    LogDebug("RPCTrigger") << "Found muonf of pt " << finalMuons[iMu].GetPtCode()
        << " L1Charge " << l1Cand.charge_packed()
        << " ql " << l1Cand.quality()
        << " fp " << finalMuons[iMu].GetFiredPlanes()
        << " b/f " << l1Cand.type_idx()
        << " phi " << phi
        << " eta " << eta
        //<< " eta l1 " << l1Cand.etaValue() // will drop out soon 
        << " killed " << finalMuons[iMu].WasKilled();
        
    
  }

  return RPCCand;
}


//define this as a plug-in
DEFINE_FWK_MODULE(RPCTrigger)
