#ifndef POPCON_RPC_GAS_SH
#define POPCON_RPC_GAS_SH

/*
 * \class RpcGasSH
 *  Core of RPC PopCon Appication
 *
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"

#include "CondFormats/RPCObjects/interface/RPCObGas.h"
#include "CondFormats/DataRecord/interface/RPCObGasRcd.h"
#include "CoralBase/TimeStamp.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "CondTools/RPC/interface/RPCFw.h"
#include<string>


namespace popcon{
  class RpcObGasData : public popcon::PopConSourceHandler<RPCObGas>{
  public:
    void getNewObjects();
    std::string id() const { return m_name;}
    ~RpcObGasData(); 
    RpcObGasData(const edm::ParameterSet& pset); 

    RPCObGas* Gasdata;

    unsigned long long snc;
    unsigned long long niov;	    
    unsigned long long utime;
  private:
    std::string m_name;
    std::string host;
    std::string user;
    std::string passw;
    unsigned long long m_since;
    unsigned long long m_till;
  };
}
#endif
