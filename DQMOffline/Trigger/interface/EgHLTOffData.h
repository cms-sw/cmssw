#ifndef DQMOFFLINE_TRIGGER_EGHLTOFFDATA
#define DQMOFFLINE_TRIGGER_EGHLTOFFDATA

//struct: EgHLTOffData (Egamma HLT Offline Data)
//
//author: Sam Harper (July 2008)
//
//WARNING: interface is NOT final, please dont use this class for now without clearing it with me
//         as I will change it and possibly break all your code
//
//aim: this is a simple struct which allows all the data needed by the egamma offline HLT DQM  code to be passed in as single object
//     this includes the TriggerEvent handle and the vector of EgHLTOffEle at the moment
//
//implimentation: 
//       
//      

#include "DQMOffline/Trigger/interface/EgHLTOffEle.h"

#include "DataFormats/Common/interface/Handle.h"

struct EgHLTOffData {
  public:
  edm::Handle<trigger::TriggerEvent> trigEvt;
  std::vector<EgHLTOffEle> *eles; //EgHLTOffEle is lightweight and handles copying well hence it isnt a vector of pointers
  std::vector<std::vector<int> > *filtersElePasses; //the filter numbers which each electron passed, the filter numbers are sorted and the electrons have an entry to entry correspondance to electrons in eles
};



#endif
