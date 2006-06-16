/*******************************************************************************
*                                                                              *
*  Karol Bunkowski                                                             *
*  Warsaw University 2003                                                      *
*                                                                              *
*******************************************************************************/
#include "L1Trigger/RPCTrigger/src/L1RpcPacManager.h"
//#include "L1Trigger/RPCTrigger/src/L1RpcException.h"                                                     
using namespace std;
/*
template<class TPacType> void L1RpcPacManager<TPacType>::Init(string patFilesDirectory) {
  PatFilesDirectory = patFilesDirectory;

  for (int tower = 0; tower < TOWER_COUNT; tower++) {
    std::vector<TPacType*> pacVec;
    PacTab.push_back(pacVec);
  }

  for (int tower = 0; tower < TOWER_COUNT; tower++) {
    TPacType* pac  = new TPacType(PatFilesDirectory, tower); //powinne byc dodane logSector i logSegment
    PacTab[tower].push_back(pac);
  }
}


//tu jest wedlug definicji logSegment 0 - 143
//powinno byc zmienione na definicje tower, logSector (0 11), logSegment (0 11)
template<class TPacType> TPacType* L1RpcPacManager<TPacType>::GetPac(int tower, int logSegmnt) {
  if (PacTab.size() <= abs(tower) )
    throw L1RpcException("L1RpcPacManager::GetPac: given towerNum to big");
   //if (PacTab[abs(towerNum)].size() <= segmentNum )
   //throw L1RpcException("L1RpcPacManager::GetPac: given segmentNum to big");

  PacTab[abs(tower)][0]->SetCurrentPosition(tower, logSegment, 0);
  return  PacTab[abs(tower)][0];
}
*/
/*
void L1RpcPacManager::FinalizePatGen(bool printStatistics) {
  for (int tower = 0; tower < TOWER_COUNT; tower++) {
    ((L1RpcPacPatGen*)PacTab[tower][0])->FinalizePatGen(printStatistics);
  }  
}
*/
