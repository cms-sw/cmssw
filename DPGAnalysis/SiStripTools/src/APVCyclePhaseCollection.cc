#include <string.h>
#include <set>
#include "DPGAnalysis/SiStripTools/interface/APVCyclePhaseCollection.h"

const int APVCyclePhaseCollection::getPhase(const std::string partition) const {

  int phase = empty;

  for(std::map<std::string,int>::const_iterator it=_apvmap.begin();it!=_apvmap.end();it++) {
    if(strstr(it->first.c_str(),partition.c_str())==it->first.c_str() || strcmp(partition.c_str(),"All")==0 ) {
      if(phase==empty ) {
	phase = it->second;
      }
      else if(phase!=it->second) {
	return multiphase;
      }
    }
  }

  if(phase==empty) return nopartition;
  return phase;


}

const std::vector<int> APVCyclePhaseCollection::getPhases(const std::string partition) const {

  std::set<int> phasesset;

  for(std::map<std::string,int>::const_iterator it=_apvmap.begin();it!=_apvmap.end();it++) {
    if(strstr(it->first.c_str(),partition.c_str())==it->first.c_str() || strcmp(partition.c_str(),"Any")==0 ) {
      if(it->second>=0 ) {
	phasesset.insert(it->second);
      }
    }
  }

  std::vector<int> phases;
  
  for(std::set<int>::const_iterator phase=phasesset.begin();phase!=phasesset.end();++phase) {
    if(*phase!=empty && *phase!=invalid) {
      phases.push_back(*phase);
    }
  }

  return phases;


}

