#ifndef POPCON_RPC_IDMAP_SH
#define POPCON_RPC_IDMAP_SH

/*
 * \class RpcIDMapSH
 *  Core of RPC PopCon Appication
 *
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include <vector>
#include <string>
#include <iostream>
#include <typeinfo>

#include "CondCore/PopCon/interface/PopConSourceHandler.h"

#include "CondFormats/RPCObjects/interface/RPCObPVSSmap.h"
#include "CondFormats/DataRecord/interface/RPCObPVSSmapRcd.h"
#include "CoralBase/TimeStamp.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "CondTools/RPC/interface/RPCFw.h"
#include<string>


namespace popcon{
  class RPCObPVSSmapData : public popcon::PopConSourceHandler<RPCObPVSSmap>{
  public:
    void getNewObjects();
    std::string id() const { return m_name;}
    ~RPCObPVSSmapData(); 
    RPCObPVSSmapData(const edm::ParameterSet& pset); 

    RPCObPVSSmap* IDMapdata;

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
