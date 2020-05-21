//---------------------------------------------------------------------------
#include "L1Trigger/RPCTrigger/interface/RPCTriggerCrate.h"
#include "L1Trigger/RPCTrigger/interface/MuonsGrabber.h"

//---------------------------------------------------------------------------
RPCTriggerCrate::RPCTriggerCrate(RPCTriggerConfiguration* triggerConfig, int tcNum)
    : m_TCGhostBusterSorter(triggerConfig) {
  //m_TCGhostBusterSorter = tcGhostBusterSorter;
  m_TriggerConfig = triggerConfig;
  m_WasMuon = false;

  m_TCNumber = tcNum;

  for (int iTB = 0; iTB < triggerConfig->getTBsInTC(); iTB++)
    m_TBsVec.push_back(RPCTriggerBoard(triggerConfig, iTB, tcNum));
}
//----------------------------------------
L1RpcTBMuonsVec RPCTriggerCrate::runTCGBSorter() {
  if (!m_WasMuon)
    return L1RpcTBMuonsVec();

  L1RpcTBMuonsVec2 tbMuonsVec2;
  for (auto& iTB : m_TBsVec) {
    tbMuonsVec2.push_back(iTB.runTBGB());
  }

  m_WasMuon = false;

  if (m_TriggerConfig->getDebugLevel() != 0) {
    for (auto& iTC : tbMuonsVec2) {
      for (unsigned int iTB = 0; iTB < iTC.size(); iTB++) {
#ifndef _STAND_ALONE
        //           LogDebug("RPCHwDebug") << "GB 1 " << iTB << " "
        //             <<tbMuonsVec2[iTC][iTB].printDebugInfo(m_TriggerConfig->getDebugLevel());
        MuonsGrabber::Instance().addMuon(iTC[iTB], 1, -1, -1, iTB);
#else
        std::cout << "GB 1 "
                  << "GB 1 " << iTB << " " << tbMuonsVec2[iTC][iTB].printDebugInfo(m_TriggerConfig->getDebugLevel())
                  << std::endl;
#endif  //_STAND_ALONE
      }
    }
  }

  return m_TCGhostBusterSorter.run(tbMuonsVec2);
}

/** Runs RPCTriggerBoard::runCone() for every TB. Cheks, if any non empty muons were found*/
void RPCTriggerCrate::runCone(const RPCLogCone& cone) {
  if (m_TBsVec[m_TriggerConfig->getTBNum(cone.getConeCrdnts())].runCone(cone))
    m_WasMuon = true;
}
