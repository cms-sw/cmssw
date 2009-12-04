#include "OHltRateCounter.h"

OHltRateCounter::OHltRateCounter(unsigned int size) {
  vector<int> itmp;
  for (unsigned int i=0;i<size;i++) {
    iCount.push_back(0);
    sPureCount.push_back(0);
    pureCount.push_back(0);
    prescaleCount.push_back(0);
    
    itmp.push_back(0);
  }
  for (unsigned int j=0;j<size;j++) {
    overlapCount.push_back(itmp);
  }

}

bool OHltRateCounter::isNewRunLS(int Run,int LumiBlock) {
  for (unsigned int i=0;i<runID.size();i++) {
    if (Run==runID[i] && LumiBlock==lumiSection[i])
      return false;
  }
  return true;
}

void OHltRateCounter::addRunLS(int Run,int LumiBlock) {
  runID.push_back(Run);
  lumiSection.push_back(LumiBlock);
  vector< int > vtmp;
  for (unsigned int i=0;i<iCount.size();i++) {
    vtmp.push_back(0);    
  }
  perLumiSectionCount.push_back(vtmp);
  perLumiSectionTotCount.push_back(0);
}


int OHltRateCounter::getIDofRunLSCounter(int Run,int LumiBlock) {
  for (unsigned int i=0;i<runID.size();i++) {
    if (Run==runID[i] && LumiBlock==lumiSection[i])
      return i;
  }
  return -999;
}

void OHltRateCounter::incrRunLSCount(int Run,int LumiBlock,int iTrig, int incr) {
  int id = getIDofRunLSCounter(Run,LumiBlock);
  if (id>-1) {
    perLumiSectionCount[id][iTrig] = perLumiSectionCount[id][iTrig] + incr;
  }
}

void OHltRateCounter::incrRunLSTotCount(int Run,int LumiBlock, int incr) {
  int id = getIDofRunLSCounter(Run,LumiBlock);
  if (id>-1) {
    perLumiSectionTotCount[id] = perLumiSectionTotCount[id] + incr;
  }
}

