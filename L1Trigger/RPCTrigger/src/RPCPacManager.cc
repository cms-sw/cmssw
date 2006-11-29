/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2003                                                      *
*                                                                              *
*******************************************************************************/
#include "L1Trigger/RPCTrigger/src/RPCPacManager.h"
//#include "L1Trigger/RPCTrigger/src/RPCException.h"                                                     
//using namespace std;
/*
template<class TPacType> void RPCPacManager<TPacType>::init(string patFilesDirectory) {
  PatFilesDirectory = patFilesDirectory;

  for (int m_tower = 0; m_tower < m_TOWER_COUNT; m_tower++) {
    std::vector<TPacType*> pacVec;
    m_PacTab.push_back(pacVec);
  }

  for (int m_tower = 0; m_tower < m_TOWER_COUNT; m_tower++) {
    TPacType* pac  = new TPacType(PatFilesDirectory, m_tower); //powinne byc dodane logSector i logSegment
    m_PacTab[m_tower].push_back(pac);
  }
}


//tu jest wedlug definicji logSegment 0 - 143
//powinno byc zmienione na definicje m_tower, logSector (0 11), logSegment (0 11)
template<class TPacType> TPacType* RPCPacManager<TPacType>::getPac(int m_tower, int logSegmnt) {
  if (m_PacTab.size() <= abs(m_tower) )
    throw RPCException("RPCPacManager::getPac: given towerNum to big");
   //if (m_PacTab[abs(towerNum)].size() <= segmentNum )
   //throw RPCException("RPCPacManager::getPac: given segmentNum to big");

  m_PacTab[abs(m_tower)][0]->setCurrentPosition(m_tower, logSegment, 0);
  return  m_PacTab[abs(m_tower)][0];
}
*/
/*
void RPCPacManager::FinalizePatGen(bool printStatistics) {
  for (int m_tower = 0; m_tower < m_TOWER_COUNT; m_tower++) {
    ((L1RpcPacPatGen*)m_PacTab[m_tower][0])->FinalizePatGen(printStatistics);
  }  
}
*/
