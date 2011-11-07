#ifndef POPCON_RPC_DATA_SRC_NOISESTRIPS_H
#define POPCON_RPC_DATA_SRC_NOISESTRIPS_H

/*
 * \class RPCNoiseStripsSH
 *  Core of RPC PopCon Appication
 *
 *  \author 
 */

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondTools/RPC/interface/RPCDBCom.h"
#include "CondFormats/RPCObjects/interface/RPCNoiseStripsObject.h"
#include "CondFormats/DataRecord/interface/RPCNoiseStripsObjectRcd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include<string>

namespace popcon{
  class RPCNoiseStripsSH : public RPCDBCom, public popcon::PopConSourceHandler<RPCNoiseStripsObject>{
  public:
    void getNewObjects();
    std::string id() const { return m_host;}
    ~RPCNoiseStripsSH(); 
    RPCNoiseStripsSH(const edm::ParameterSet& pset); 
    void run(){};
    RPCNoiseStripsObject * rpcNoiseStrips;

  private:
    std::string m_host;
    std::string m_user;
    std::string m_passw;
    
    bool m_first;
    unsigned int m_version;
    unsigned int m_run;

  };
}
#endif

