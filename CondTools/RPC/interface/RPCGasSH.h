#ifndef POPCON_RPC_GAS_SH
#define POPCON_RPC_GAS_SH

/*
 * \class RpcGasSH
 *  Core of RPC PopCon Appication
 *
 *  $Date: 2008/07/31 14:50:15 $
 *  $Revision: 1.1 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"

#include "CondFormats/RPCObjects/interface/RPCGas.h"
#include "CondFormats/DataRecord/interface/RPCGasRcd.h"
#include "CoralBase/TimeStamp.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "CondTools/RPC/interface/RPCFw.h"
#include<string>


namespace popcon{
  class RpcGasData : public popcon::PopConSourceHandler<RPCGas>{
  public:
    void getNewObjects();
    std::string id() const { return m_name;}
    ~RpcGasData(); 
    RpcGasData(const edm::ParameterSet& pset); 

    RPCGas* Gasdata;
    RPCGas* Tdata;

    int snc;
    int niov;	    
    int utime;
  private:
    std::string m_name;
    std::string host;
    std::string user;
    std::string passw;
    std::string Ohost;
    std::string Ouser;
    std::string Opassw;
    unsigned long long m_since;

  };
}
#endif
