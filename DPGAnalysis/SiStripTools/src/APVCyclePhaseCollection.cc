#include "DPGAnalysis/SiStripTools/interface/APVCyclePhaseCollection.h"

const int APVCyclePhaseCollection::getPhase(const std::string partition) const {

  int phase = invalid;

  for(std::map<std::string,int>::const_iterator it=_apvmap.begin();it!=_apvmap.end();it++) {
    if(strstr(it->first.c_str(),partition.c_str())==it->first.c_str() || strcmp(partition.c_str(),"All")==0 ) {
      if(phase==invalid ) {
	phase = it->second;
      }
      else if(phase!=it->second) {
	return multiphase;
      }
    }
  }

  if(phase==invalid) return nopartition;
  return phase;


}
