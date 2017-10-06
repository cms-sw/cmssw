#ifndef POPCON_RPC_DATA_SRC_H
#define POPCON_RPC_DATA_SRC_H

/*
 * \class RpcData
 *  Core of RPC PopCon Appication
 *
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"

#include "CondFormats/RPCObjects/interface/RPCObFebmap.h"
#include "CondFormats/DataRecord/interface/RPCObFebmapRcd.h"
#include "CoralBase/TimeStamp.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "CondTools/RPC/interface/RPCFw.h"
#include<string>


namespace popcon{
  class RpcDataFebmap : public popcon::PopConSourceHandler<RPCObFebmap>{
  public:
    void getNewObjects() override;
    std::string id() const override { return m_name;}
    ~RpcDataFebmap() override; 
    RpcDataFebmap(const edm::ParameterSet& pset); 

    RPCObFebmap* Febdata;

    unsigned long long snc;
    unsigned long long tll;
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

