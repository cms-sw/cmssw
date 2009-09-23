#ifndef POPCON_RPC_DATA_SRC_H
#define POPCON_RPC_DATA_SRC_H

/*
 * \class RPCImonSH
 *  Core of RPC PopCon Appication
 *
 *  $Date: 2008/12/12 20:02:50 $
 *  $Revision: 1.3 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"

#include "CondFormats/RPCObjects/interface/RPCObCond.h"
#include "CondFormats/DataRecord/interface/RPCObCondRcd.h"
#include "CoralBase/TimeStamp.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "CondTools/RPC/interface/RPCFw.h"
#include<string>


namespace popcon{
  class RpcDataI : public popcon::PopConSourceHandler<RPCObImon>{
  public:
    void getNewObjects();
    std::string id() const { return m_name;}
    ~RpcDataI(); 
    RpcDataI(const edm::ParameterSet& pset); 

    RPCObImon* Idata;

    unsigned long long snc;
    unsigned long long niov;	    
    unsigned long long utime;
  private:
    std::string m_name;
    std::string host;
    std::string user;
    std::string passw;
    unsigned long long m_since;

  };
}
#endif

