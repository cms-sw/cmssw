/** \file RPCTrigger.cc
 *
 *  $Date: 2006/08/28 13:06:49 $
 *  $Revision: 1.18 $
 *  \author Tomasz Fruboes
 */
#include "L1Trigger/RPCTrigger/interface/RPCTrigger.h"
#include <FWCore/ParameterSet/interface/FileInPath.h>

//#define ML_DEBUG 





RPCTrigger::RPCTrigger(const edm::ParameterSet& iConfig)
{
  produces<std::vector<L1MuRegionalCand> >("RPCb");
  produces<std::vector<L1MuRegionalCand> >("RPCf");
  
  std::string patternsDirNameLocal = iConfig.getParameter<std::string>("RPCPatternsDir");
  //std::string patternsDirName = patternsDirNameLocal;


  // Since fileInPath doesnt allow us to use directory we use this quick and dirty solution
  edm::FileInPath fp(patternsDirNameLocal+"keepme.txt"); 
  std::string patternsDirNameUnstriped = fp.fullPath();
  std::string patternsDirName = patternsDirNameUnstriped.substr(0,patternsDirNameUnstriped.find_last_of("/")+1);

  /*
  const char * rb = ::getenv("CMSSW_RELEASE_BASE"); 
  std::string patternsDirName(rb);
  std::cout << std::endl << patternsDirName << std::endl;
  patternsDirName+="/src/"+patternsDirNameLocal;
  std::cout << std::endl << patternsDirName << std::endl;*/

  int triggerDebug = iConfig.getUntrackedParameter("RPCTriggerDebug",0);
  
  // 0 - no debug
  // 2 - technical debug
  // 1 - human readable debug
  if ( triggerDebug != 1 && triggerDebug != 2)
     triggerDebug = 0;
        
  m_pacManager.Init(patternsDirName, _12_PACS_PER_TOWER);
  
  m_trigConfig = new L1RpcBasicTrigConfig(&m_pacManager);
  
  m_trigConfig->SetDebugLevel(triggerDebug);
  
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
//  iEvent.getByType(rpcDigis);
  iEvent.getByLabel("muonRPCDigis",rpcDigis);

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

    if (finalMuons[iMu].GetPtCode()==0){
      continue; 
    } 

    L1MuRegionalCand l1Cand;
    
    
    l1Cand.setQualityPacked(finalMuons[iMu].GetQuality());
    l1Cand.setPtPacked(finalMuons[iMu].GetPtCode());
    
    l1Cand.setType(type); 
    
    int charge=finalMuons[iMu].GetSign();
    
    if (charge == 0)  // negative
      l1Cand.setChargePacked(1);
    else  
      l1Cand.setChargePacked(0);
    
    //L1RpcConst::L1RpcConeCrdnts cone = finalMuons[iMu].GetConeCrdnts();    
    
    /*
    int pac = cone.LogSector*12+cone.LogSegment;
    const float pi = 3.14159265;
    const float offset = 5*(2*pi/360); // redefinition! Defined also in RPCRingFromRolls::phiMapCompare
    float phi = 2*pi*pac/144-offset;
    if (phi<0)
      phi+=2*pi;
    
    l1Cand.setPhiValue(phi);
    */

    //Note: pac numbering begins at 5 deg and goes from 1 to 144.
    // we want phi values from 0 to 2.5 deg to be phiPacked=0 
    // max phiPacked value is 143 (see CMS IN 2004-022)
    int phiPacked = (finalMuons[iMu].GetPhiAddr()+2)%144;
    l1Cand.setPhiPacked(phiPacked);
/*
    float eta = L1RpcConst::etaFromTowerNum(cone.Tower);
    l1Cand.setEtaValue(eta);
*/
    //Note: etaAddr is packed in special way: see CMS IN 2004-022
    signed short etaAddr = finalMuons[iMu].GetEtaAddr()-16; // -16..16
    bool etaNegative = false;
    if (etaAddr < 0){
      etaNegative = true;
      etaAddr = ~(-etaAddr)+1; // convert to negative :)
    }

    etaAddr &= 63; // 6 bits only
         
    l1Cand.setEtaPacked(etaAddr);

    /*    
    std::cout<< std::endl << "RBMuon::" << finalMuons[iMu].GetEtaAddr() << " " 
             << finalMuons[iMu].GetPhiAddr() << std::endl ;
    std::cout<< "cand " <<  l1Cand.eta_packed() << " " 
             << l1Cand.phi_packed() << std::endl ;
    //*/

    RPCCand.push_back(l1Cand);
        
    LogDebug("RPCTrigger") << "Found muonf of pt " << finalMuons[iMu].GetPtCode()
        << " L1Charge " << l1Cand.charge_packed()
        << " ql " << l1Cand.quality()
        << " fp " << finalMuons[iMu].GetFiredPlanes()
        << " b/f " << l1Cand.type_idx()
        << " phi " <<  l1Cand.phi_packed()
        << " eta " << l1Cand.eta_packed()
        //<< " eta l1 " << l1Cand.etaValue() // will drop out soon 
        << " killed " << finalMuons[iMu].WasKilled();
        
    
  }

  return RPCCand;
}


//define this as a plug-in
DEFINE_FWK_MODULE(RPCTrigger)
