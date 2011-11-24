#ifndef POPCON_RPC_DATA_SRC_H
#define POPCON_RPC_DATA_SRC_H

/*
 * \class DoAQuerySH
 *  Core of RPC PopCon Appication
 *
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */

#include "CondCore/PopCon/interface/PopConSourceHandler.h"
#include "CondTools/RPC/interface/RPCDBCom.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include<string>

namespace popcon{
  class DoAQuerySH : public RPCDBCom, public popcon::PopConSourceHandler<int>{
  public:
    void getNewObjects();
    std::string id() const { return m_host;}
    ~DoAQuerySH(); 
    DoAQuerySH(const edm::ParameterSet& pset); 
    void run(){};

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

