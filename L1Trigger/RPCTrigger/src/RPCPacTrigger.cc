/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2004                                                      *
*                                                                              *
*******************************************************************************/
#include "L1Trigger/RPCTrigger/src/RPCPacTrigger.h"
#ifdef _GRAB_MUONS
#include "PactAnalysis/L1RpcMuonsGrabber.h"
#endif

RPCPacTrigger::RPCPacTrigger(RPCTriggerConfiguration* triggerConfig):
  m_FinalSorter(triggerConfig), m_TCGhostBusterSorter(triggerConfig), m_TBGhostBuster() {
  m_TrigCnfg = triggerConfig;
  for(int iTC = 0; iTC < m_TrigCnfg->getTCsCnt(); iTC++) {
    m_TriggerCratesVec.push_back(RPCTriggerCrate(&m_TCGhostBusterSorter, &m_TBGhostBuster, m_TrigCnfg, iTC) );
  }
}

////////////////////////////////////////////////////////////////////////////////////

L1RpcTBMuonsVec2 RPCPacTrigger::runEvent(const L1RpcLogConesVec& logConesVec) {
  m_GBFinalMuons.clear();

 if (m_TrigCnfg->getDebugLevel()!=0){
#ifdef _STAND_ALONE
   std::cout << "---TBMuons in new event" << std::endl;
#else
   LogDebug("RPCHwDebug") << "---TBMuons in new event";
#endif // _STAND_ALONE
 }
  for(unsigned int iLC = 0; iLC < logConesVec.size(); iLC++) {
    if(logConesVec[iLC].getFiredPlanesCnt() >= 3) {
      m_TriggerCratesVec[m_TrigCnfg->getTCNum(logConesVec[iLC].getConeCrdnts())].runCone(logConesVec[iLC]);
    }
  }

  L1RpcTBMuonsVec2 tcsMuonsVec2;
  for(unsigned int iTC = 0; iTC < m_TriggerCratesVec.size(); iTC++) {
    tcsMuonsVec2.push_back(m_TriggerCratesVec[iTC].runTCGBSorter() );
  }
      
  if (m_TrigCnfg->getDebugLevel()!=0){
    for (unsigned  int iTC = 0; iTC < tcsMuonsVec2.size(); iTC++){
        for (unsigned  int iTB = 0; iTB < tcsMuonsVec2[iTC].size(); iTB++){
#ifdef _STAND_ALONE
	  std::cout << "GB 2 " <<tcsMuonsVec2[iTC][iTB].printDebugInfo(m_TrigCnfg->getDebugLevel()) << std::endl;
#else
	  LogDebug("RPCHwDebug") << "GB 2 " <<tcsMuonsVec2[iTC][iTB].printDebugInfo(m_TrigCnfg->getDebugLevel());
#endif // _STAND_ALONE
        }
    }
  }

  m_GBFinalMuons = m_FinalSorter.run(tcsMuonsVec2);
  /* // Moved to m_FinalSorter to have HS and FS info
  if (m_TrigCnfg->getDebugLevel()!=0){
     // iterate over m_GBFinalMuons and call printDebug()
    for (unsigned  int iTC = 0; iTC < m_GBFinalMuons.size(); iTC++){
        for (unsigned  int iTB = 0; iTB < m_GBFinalMuons[iTC].size(); iTB++){
#ifdef _STAND_ALONE
	  std::cout <<"GB 3 "<<m_GBFinalMuons[iTC][iTB].printDebugInfo(m_TrigCnfg->getDebugLevel())<< std::endl;
#else
	  LogDebug("RPCHwDebug") <<"GB 3 "<<m_GBFinalMuons[iTC][iTB].printDebugInfo(m_TrigCnfg->getDebugLevel());

#endif // _STAND_ALONE
        }
    }                               
  }
  */
  /*
  #ifdef _GRAB_MUONS
    L1RpcMuonsGrabber::Instance()->StoreAnswers(m_GBFinalMuons);
  #endif
  */
  
#ifdef GETCONES
  bool foundMuons = false;
  L1RpcTBMuonsVec bMuons = m_GBFinalMuons[0];
  L1RpcTBMuonsVec fMuons = m_GBFinalMuons[1];

  std::cout<< "------------" << std::endl;
  for (unsigned int i = 0; i < bMuons.size(); i++){
    if (bMuons[i].getPtCode()!=0){ 
       /*
       std::cout<< "ptcode " << bMuons[i].getPtCode() 
                << " t " <<  bMuons[i].getTower()
                << " sec " <<  bMuons[i].getLogSector()
                << " seg " <<  bMuons[i].getLogSegment()
                << std::endl;*/
       foundMuons = true;
   }
  }
  for (unsigned int i = 0; i < fMuons.size(); i++){
    if (fMuons[i].getPtCode()!=0){
       /*std::cout<< "ptcode " << fMuons[i].getPtCode()
                << " t " <<  fMuons[i].getTower()
                << " sec " <<  fMuons[i].getLogSector()
                << " seg " <<  fMuons[i].getLogSegment()
                << std::endl;*/
       foundMuons = true;
       }
  }
  if (!foundMuons){
    for(unsigned int iLC = 0; iLC < logConesVec.size(); iLC++) {
      if(logConesVec[iLC].getFiredPlanesCnt() >= 3) {
        std::cout<< logConesVec[iLC].toString();
      }
    }
  }
#endif



  return m_GBFinalMuons;
}

////////////////////////////////////////////////////////////////////////////////////

L1RpcTBMuonsVec RPCPacTrigger::getNotEmptyMuons()  {
  L1RpcTBMuonsVec notEmptyMuonsVec;
  for(unsigned int iMu = 0; iMu < m_GBFinalMuons[0].size(); iMu++)
    if(m_GBFinalMuons[0][iMu].getCode() != 0)
      notEmptyMuonsVec.push_back(m_GBFinalMuons[0][iMu]);

  for(unsigned int iMu = 0; iMu < m_GBFinalMuons[1].size(); iMu++)
    if(m_GBFinalMuons[1][iMu].getCode() != 0)
      notEmptyMuonsVec.push_back(m_GBFinalMuons[1][iMu]);

  return notEmptyMuonsVec;
}
