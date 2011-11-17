#ifndef POPCON_RPC_DATA_SRC_H
#define POPCON_RPC_DATA_SRC_H

/*
 * \class RpcData
 *  Core of RPC PopCon Appication
 *
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "CondCore/PopCon/interface/PopConSourceHandler.h"

#include "CondFormats/RPCObjects/interface/RPCObCond.h"
#include "CondFormats/DataRecord/interface/RPCObCondRcd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondTools/RPC/interface/RPCFw.h"
#include<string>

namespace popcon{
  class RpcDataV : public popcon::PopConSourceHandler<RPCObVmon>{
  public:
    void getNewObjects();
    std::string id() const { return m_name;}
    ~RpcDataV(); 
    RpcDataV(const edm::ParameterSet& pset); 

    RPCObVmon* Vdata;

  private:
    std::string m_name;
    std::string host;
    std::string user;
    std::string passw;

    bool m_first;
    int m_test_suptype;
    unsigned long long m_since;
    unsigned long long m_range;
    int n_IOVmax;
  };
}
#endif

