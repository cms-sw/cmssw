#include "L1Trigger/RPCTrigger/interface/RPCTrigger.h"

// Configuration via eventsetup:
#include "CondFormats/DataRecord/interface/L1RPCConfigRcd.h"
#include "CondFormats/L1TObjects/interface/L1RPCConfig.h"

#include "CondFormats/DataRecord/interface/L1RPCConeBuilderRcd.h"
#include "CondFormats/RPCObjects/interface/L1RPCConeBuilder.h"

#include "CondFormats/DataRecord/interface/L1RPCHwConfigRcd.h"
#include "CondFormats/RPCObjects/interface/L1RPCHwConfig.h"

#include "CondFormats/DataRecord/interface/L1RPCBxOrConfigRcd.h"
#include "CondFormats/L1TObjects/interface/L1RPCBxOrConfig.h"



//#define ML_DEBUG 



#include "L1Trigger/RPCTrigger/interface/MuonsGrabber.h"


RPCTrigger::RPCTrigger(const edm::ParameterSet& iConfig):
   m_trigConfig(0),m_pacTrigger(0)
{
  produces<std::vector<L1MuRegionalCand> >("RPCb");
  produces<std::vector<L1MuRegionalCand> >("RPCf");

  produces<std::vector<RPCDigiL1Link> >("RPCb");
  produces<std::vector<RPCDigiL1Link> >("RPCf");

  m_firstRun = true;
  m_cacheID = 0;


  m_triggerDebug = iConfig.getUntrackedParameter<int>("RPCTriggerDebug",0);
  
  // 0 - no debug
  // 1 - dump to xml
  if ( m_triggerDebug != 1){
     m_triggerDebug = 0;
  }
     
   
  m_label = iConfig.getParameter<std::string>("label");
}




RPCTrigger::~RPCTrigger(){ 
   if (m_pacTrigger != 0) delete m_pacTrigger;
   if (m_trigConfig != 0) delete m_trigConfig;
}



void
RPCTrigger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
 
  //static int ev=1;
  //std::cout << "----------------------------------- "  << ev++ << std::endl;
  if ( m_triggerDebug == 1)  MuonsGrabber::Instance().startNewEvent(iEvent.id().event(), iEvent.bunchCrossing());  

  if (m_firstRun){

     m_cacheID = iSetup.get<L1RPCConfigRcd>().cacheIdentifier();
     m_firstRun = false;  
     edm::ESHandle<L1RPCConfig> conf;
     iSetup.get<L1RPCConfigRcd>().get(conf);
     const L1RPCConfig *rpcconf = conf.product();


     m_pacManager.init(rpcconf);
     m_trigConfig = new RPCBasicTrigConfig(&m_pacManager);
     m_trigConfig->setDebugLevel(m_triggerDebug);
     m_pacTrigger = new RPCPacTrigger(m_trigConfig);
     if ( m_triggerDebug == 1)  MuonsGrabber::Instance().setRPCBasicTrigConfig(m_trigConfig);

      
  }

  if (m_cacheID != iSetup.get<L1RPCConfigRcd>().cacheIdentifier()) {

      //std::cout << " New pats: " << iSetup.get<L1RPCConfigRcd>().cacheIdentifier() << std::endl ; 
      m_cacheID = iSetup.get<L1RPCConfigRcd>().cacheIdentifier();

     edm::ESHandle<L1RPCConfig> conf;
     iSetup.get<L1RPCConfigRcd>().get(conf);
     const L1RPCConfig *rpcconf = conf.product();

     m_pacManager.init(rpcconf);
     delete m_trigConfig;
     m_trigConfig = new RPCBasicTrigConfig(&m_pacManager);
     m_trigConfig->setDebugLevel(m_triggerDebug);

     delete m_pacTrigger;
     m_pacTrigger = new RPCPacTrigger(m_trigConfig);

     if ( m_triggerDebug == 1)  MuonsGrabber::Instance().setRPCBasicTrigConfig(m_trigConfig);

       
  }

 
  
  edm::Handle<RPCDigiCollection> rpcDigis;
//  iEvent.getByType(rpcDigis);
  //iEvent.getByLabel("muonRPCDigis",rpcDigis);
  iEvent.getByLabel(m_label, rpcDigis);

  std::auto_ptr<std::vector<L1MuRegionalCand> > candBarell(new std::vector<L1MuRegionalCand>);
  std::auto_ptr<std::vector<L1MuRegionalCand> > candForward(new std::vector<L1MuRegionalCand>);
  if (!rpcDigis.isValid()) 
  {
     LogDebug("RPCTrigger")
          << "\nWarning: RPCDigiCollection with input tag " << m_label
          << "\nrequested in configuration, but not found in the event. Emulator will produce empty collection \n ";

          iEvent.put(candBarell, "RPCb");
          iEvent.put(candForward, "RPCf");
 
          return;
  }

  
  if (rpcDigis->begin() == rpcDigis->end() )
  {
     LogDebug("RPCTrigger")
          << "\nWarning: RPCDigiCollection with input tag " << m_label
          << "\n seems to be empty for this event. Emulator will run on empty collection ";

  }



  std::auto_ptr<std::vector<RPCDigiL1Link> >  brlLinks(new std::vector<RPCDigiL1Link>);
  std::auto_ptr<std::vector<RPCDigiL1Link> >  fwdLinks(new std::vector<RPCDigiL1Link>);
    
  for (int iBx = -1; iBx < 2; ++ iBx) {
    
    L1RpcLogConesVec ActiveCones;

    edm::ESHandle<L1RPCConeBuilder> coneBuilder;
    iSetup.get<L1RPCConeBuilderRcd>().get(coneBuilder);
      
    edm::ESHandle<L1RPCConeDefinition> l1RPCConeDefinition;
    iSetup.get<L1RPCConeDefinitionRcd>().get(l1RPCConeDefinition);

    edm::ESHandle<L1RPCHwConfig> hwConfig;
    iSetup.get<L1RPCHwConfigRcd>().get(hwConfig);

    edm::ESHandle<L1RPCBxOrConfig> bxOrConfig;
    iSetup.get<L1RPCBxOrConfigRcd>().get(bxOrConfig);



    ActiveCones = m_theLinksystemFromES.getConesFromES(rpcDigis, coneBuilder, l1RPCConeDefinition, bxOrConfig, hwConfig, iBx);
    
    edm::ESHandle<L1RPCHsbConfig> hsbConfig;
    iSetup.get<L1RPCHsbConfigRcd>().get(hsbConfig);

    L1RpcTBMuonsVec2 finalMuons = m_pacTrigger->runEvent(ActiveCones, hsbConfig);
  
    //int maxFiredPlanes = 0;
    
    /*
    for (unsigned int i=0;i<ActiveCones.size();i++){
        int fpCnt = ActiveCones[i].getFiredPlanesCnt();
        if (fpCnt > maxFiredPlanes)
          maxFiredPlanes = fpCnt;
    }
  
    // Fill out the products
    // finalMuons[0]=barell, finalMuons[1]=endcap
    LogDebug("RPCTrigger") << "---Filling candindates in new event--- " 
                          << maxFiredPlanes << std::endl;
<<<<<<< RPCTrigger.cc
    */
    
//    std::vector<L1MuRegionalCand> RPCb = giveFinallCandindates(finalMuons[0],1, iBx);
//    std::vector<L1MuRegionalCand> RPCf = giveFinallCandindates(finalMuons[1],3, iBx);
    std::vector<RPCDigiL1Link> dlBrl;  
    std::vector<RPCDigiL1Link> dlFwd;  
    std::vector<L1MuRegionalCand> RPCb = giveFinallCandindates(finalMuons[0],1, iBx, rpcDigis, dlBrl);
    std::vector<L1MuRegionalCand> RPCf = giveFinallCandindates(finalMuons[1],3, iBx, rpcDigis, dlFwd);
      
    
    brlLinks->insert(brlLinks->end(), dlBrl.begin(), dlBrl.end() );
    fwdLinks->insert(fwdLinks->end(), dlFwd.begin(), dlFwd.end() );

    candBarell->insert(candBarell->end(), RPCb.begin(), RPCb.end());
    candForward->insert(candForward->end(), RPCf.begin(), RPCf.end());

    if ( m_triggerDebug == 1)  MuonsGrabber::Instance().writeDataForRelativeBX(iBx);  
  }  

  iEvent.put(fwdLinks, "RPCf");
  iEvent.put(brlLinks, "RPCb");
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
std::vector<L1MuRegionalCand> RPCTrigger::giveFinallCandindates(const L1RpcTBMuonsVec& finalMuons, int type, int bx,   
                                     edm::Handle<RPCDigiCollection> rpcDigis, std::vector<RPCDigiL1Link> & retRPCDigiLink)
{

  std::vector<L1MuRegionalCand> RPCCand;

  for(unsigned int iMu = 0; iMu < finalMuons.size(); iMu++)
  {

    if (finalMuons[iMu].getPtCode()==0){
      continue; 
    } 

    RPCDigiL1Link newDigiLink;

    //std::cout << "######################################## " << std::endl;
    //std::cout << finalMuons[iMu].getPhiAddr() << " " << finalMuons[iMu].getEtaAddr() << std::endl;
    RPCMuon::TDigiLinkVec digiIVec = finalMuons[iMu].getDigiIdxVec();
    // Here the iteration has to be the same as in 
    short int digiIndex = 0;
    RPCDigiCollection::DigiRangeIterator detUnitIt;
    for (detUnitIt=rpcDigis->begin();
         detUnitIt!=rpcDigis->end();
         ++detUnitIt)
    {

      const RPCDetId& id = (*detUnitIt).first;
      uint32_t rawId = id.rawId();
      const RPCDigiCollection::Range& range = (*detUnitIt).second;


      for (RPCDigiCollection::const_iterator digiIt = range.first;
           digiIt!=range.second;
           ++digiIt)
      {
        ++digiIndex;

        RPCMuon::TDigiLinkVec::iterator it = digiIVec.begin();
        for(;it!=digiIVec.end();++it) {
           if (digiIndex==it->m_digiIdx) {
             newDigiLink.setLink(it->m_layer+1, rawId, digiIt->strip(), digiIt->bx() );
             //std::cout << type << " " << iMu << " layer: " << it->m_layer <<  " index " << it->m_digiIdx << std::endl;
             //std::cout << "         " << id << " " << " |bx " << digiIt->bx() << " strip " << digiIt->strip() << std::endl;  
           }
        }
      }
    }
    retRPCDigiLink.push_back(newDigiLink);


    L1MuRegionalCand l1Cand;
    
    l1Cand.setBx(bx);
    
    
    l1Cand.setQualityPacked(finalMuons[iMu].getQuality());
    l1Cand.setPtPacked(finalMuons[iMu].getPtCode());
    
    l1Cand.setType(type); 
    
    int charge=finalMuons[iMu].getSign();
    
    if (charge == 0)  // negative
      l1Cand.setChargePacked(1);
    else  
      l1Cand.setChargePacked(0);
    
    //RPCConst::l1RpcConeCrdnts cone = finalMuons[iMu].getConeCrdnts();    
    
    /*
    int pac = cone.m_LogSector*12+cone.m_LogSegment;
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
    //int phiPacked = (finalMuons[iMu].getPhiAddr()+2)%144;
    int phiPacked = finalMuons[iMu].getPhiAddr();
    l1Cand.setPhiPacked(phiPacked);
/*
    float eta = RPCConst::etaFromTowerNum(cone.m_Tower);
    l1Cand.setEtaValue(eta);
*/
    //Note: etaAddr is packed in special way: see CMS IN 2004-022
    signed short etaAddr = finalMuons[iMu].getEtaAddr(); // 
//    signed short etaAddr = finalMuons[iMu].getEtaAddr()-16; // -16..16
//    bool etaNegative = false;
//    if (etaAddr < 0){
//      etaNegative = true;
//      etaAddr = ~(-etaAddr)+1; // convert to negative :)
//    }

//    etaAddr &= 63; // 6 bits only
         
    l1Cand.setEtaPacked(etaAddr);
    l1Cand.setChargeValid(true);

    /*    
    std::cout<< std::endl << "RBMuon::" << finalMuons[iMu].getEtaAddr() << " " 
             << finalMuons[iMu].getPhiAddr() << std::endl ;
    std::cout<< "cand " <<  l1Cand.eta_packed() << " " 
             << l1Cand.phi_packed() << std::endl ;
    */

    RPCCand.push_back(l1Cand);
        
    LogDebug("RPCTrigger") << "Found muonf of pt " 
        << finalMuons[iMu].getPtCode()
        << " bx " << l1Cand.bx()
        << " L1Charge " << l1Cand.charge_packed()
        << " ql " << l1Cand.quality()
        << " fp " << finalMuons[iMu].getFiredPlanes()
        << " b/f " << l1Cand.type_idx()
        << " phi " <<  l1Cand.phi_packed()
        << " eta " << l1Cand.eta_packed()
        //<< " eta l1 " << l1Cand.etaValue() // will drop out soon 
        << " killed " << finalMuons[iMu].wasKilled();
        
    
  }

  return RPCCand;
}

