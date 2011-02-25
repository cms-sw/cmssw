#ifndef RPCDBClsPerformanceHandler_H
#define RPCDBClsPerformanceHandler_H

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CondFormats/RPCObjects/interface/RPCClusterSize.h"
#include "CondFormats/DataRecord/interface/RPCClusterSizeRcd.h"

//#include "CondFormats/RPCObjects/interface/RPCStripNoises.h"
//#include "CondFormats/DataRecord/interface/RPCStripNoisesRcd.h"


#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"

class  RPCDBClsSimSetUp;


  class RPCDBClsPerformanceHandler: public popcon::PopConSourceHandler<RPCClusterSize>{
  public:
    void getNewObjects();
    ~RPCDBClsPerformanceHandler(); 
    RPCDBClsPerformanceHandler(const edm::ParameterSet& pset); 
    std::string id() const;

  private:
 
    unsigned long long m_since;
    std::string dataTag;
       RPCDBClsSimSetUp* theRPCClsSimSetUp;
  };

#endif 
