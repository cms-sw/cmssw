#ifndef RPC_DB_FW_H
#define RPC_DB_FW_H

/*
 * \class RPCFw
 *  Reads data from OMDS and creates conditioning objects
 *
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */



#include "CondTools/RPC/interface/RPCDBCom.h"
#include "CoralBase/TimeStamp.h"
#include "CondTools/RPC/interface/RPCImonSH.h"
#include "CondTools/RPC/interface/RPCVmonSH.h"
#include "CondTools/RPC/interface/RPCStatusSH.h"
#include "CondTools/RPC/interface/RPCTempSH.h"
#include "CondTools/RPC/interface/RPCGasSH.h"
#include "CondTools/RPC/interface/RPCIDMapSH.h"
#include "CondFormats/RPCObjects/interface/RPCObCond.h"
#include "CondFormats/RPCObjects/interface/RPCObFebmap.h"
#include "CondFormats/RPCObjects/interface/RPCObUXC.h"
#include "CondFormats/RPCObjects/interface/RPCObGasMix.h"

struct dbread{
    float alias;
    float value;
};


class RPCFw : virtual public RPCDBCom
{
public:
  RPCFw( const std::string& connectionString,
         const std::string& userName,
         const std::string& password);
  virtual ~RPCFw();
  void run();

  coral::TimeStamp UTtoCT(long long utime);
  unsigned int CTtoUT(const coral::TimeStamp& time);

  std::vector<RPCObImon::I_Item> createIMON(long long since, long long till);
  std::vector<RPCObVmon::V_Item> createVMON(long long from, long long till); 
  std::vector<RPCObStatus::S_Item> createSTATUS(long long since, long long till); 
  std::vector<RPCObGas::Item> createGAS(long long since, long long till);
  std::vector<RPCObTemp::T_Item> createT(long long since, long long till);
  std::vector<RPCObPVSSmap::Item> createIDMAP();
  std::vector<RPCObFebmap::Feb_Item> createFEB(long long since, long long till);	
  std::vector<RPCObUXC::Item> createUXC(long long since, long long till);
  std::vector<RPCObGasMix::Item> createMix(long long since, long long till);
  void setSuptype(int test_suptype);

 private:  
  unsigned int detId(int pvssId);
  
private:
  coral::TimeStamp tMIN;
  coral::TimeStamp tMAX;
  std::string m_connectionString;
  std::string m_userName;
  std::string m_password;
 private:
  //  int _pvssId;
  std::map<int, unsigned int> pvssTodetId;
  int _suptype;
};

#endif
