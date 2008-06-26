#ifndef POPCON_RPC_GAS_DATA_SRC_H
#define POPCON_RPC_GAS_DATA_SRC_H

/*
 * \class RpcData
 *  Core of RPC Gas PopCon Appication
 *
 *  $Date: 2008/05/01 15:27:10 $
 *  $Revision: 1.4 $
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
#include "RPCFw.h"
#include "TimeConv.h"
#include<string>

namespace popcon{
  class RpcGas : public popcon::PopConSourceHandler<RPCGas>{
  public:
    void getNewObjects();
    std::string id() const { return m_name;}
    ~RpcGas(); 
    RpcGas(const edm::ParameterSet& pset); 
    void writelast(int newtime);

    RPCGas* Gdata;
    RPCGas* Tdata;

    int snc;
    int tll;	    
    	

  private:
    std::string m_name;
    std::string host;
    std::string user;
    std::string passw;
    std::string Ohost;
    std::string Ouser;
    std::string Opassw;
    int since;
    std::string logpath;
  };
}
#endif
