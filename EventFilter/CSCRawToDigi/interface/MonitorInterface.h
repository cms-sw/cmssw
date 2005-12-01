#ifndef MonitorInterface_h
#define MonitorInterface_h
#include "FWCore/ParameterSet/interface/ParameterSet.h"
class CSCDCCEventData;

class MonitorInterface{

public:

//    MonitorInterface(edm::ParameterSet const &pset){}
    MonitorInterface(){}
    virtual ~MonitorInterface(){}
    virtual void process(CSCDCCEventData & dccData){}

      

private:


};

#endif
