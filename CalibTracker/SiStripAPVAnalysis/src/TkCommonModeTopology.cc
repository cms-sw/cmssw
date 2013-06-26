#include "CalibTracker/SiStripAPVAnalysis/interface/TkCommonModeTopology.h"

int TkCommonModeTopology::setOfStrip(int in) {
  for (int i = 0; i<numberOfSets(); i++){
    if (in >=initialStrips()[i] && in <= finalStrips()[i]) return i;
  }
  return -1;
}

TkCommonModeTopology::TkCommonModeTopology(int nstrips, int nstripsperset) : numberStrips(nstrips), 
  numberStripsPerSet(nstripsperset){
  initStrips.clear();
  finStrips.clear();
  for (int i=0; i<numberOfSets(); i++){
    initStrips.push_back(i*numberOfStripsPerSet());
    finStrips.push_back((i+1)*numberOfStripsPerSet()-1);
  }
}


 
