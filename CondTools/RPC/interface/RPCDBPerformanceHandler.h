#ifndef RPCDBPerformanceHandler_H
#define RPCDBPerformanceHandler_H

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"


#include "CondFormats/RPCObjects/interface/RPCStripNoises.h"
#include "CondFormats/DataRecord/interface/RPCStripNoisesRcd.h"


#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

class  RPCDBSimSetUp;


  class RPCDBPerformanceHandler: public popcon::PopConSourceHandler<RPCStripNoises>{
  public:
    void getNewObjects() override;
    ~RPCDBPerformanceHandler() override; 
    RPCDBPerformanceHandler(const edm::ParameterSet& pset); 
    std::string id() const override;

  private:
 
    unsigned long long m_since;
    std::string dataTag;
       RPCDBSimSetUp* theRPCSimSetUp;
  };

#endif 
