#ifndef RPC_DB_RE_H
#define RPC_DB_RE_H

/*
 * \class RPCIOVReader
 *  Reads data from OMDS and creates conditioning objects
 *
 *  $Date: 2009/04/13 20:39:53 $
 *  $Revision: 1.14 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */



#include "CondTools/RPC/interface/RPCDBCom.h"
#include "CoralBase/TimeStamp.h"
#include "CondTools/RPC/interface/RPCImonSH.h"
#include "CondTools/RPC/interface/RPCVmonSH.h"
#include "CondTools/RPC/interface/RPCStatusSH.h"
#include "CondTools/RPC/interface/RPCTempSH.h"
#include "CondFormats/RPCObjects/interface/RPCObCond.h"
#include "CondTools/RPC/interface/RPCGasSH.h"
#include "CondTools/RPC/interface/RPCIDMapSH.h"
#include "CondFormats/RPCObjects/interface/RPCObFebmap.h"


class RPCIOVReader : virtual public RPCDBCom
{
public:
  RPCIOVReader( const std::string& connectionString,
		const std::string& userName,
		const std::string& password);
  virtual ~RPCIOVReader();
  void run();
  
  std::vector<unsigned long long> listIOV();
  std::vector<RPCObImon::I_Item> getIMON(unsigned long long IMIN, unsigned long long IMAX);

  std::string toDay(int intday);
  std::string toTime(int inttime);
  
private:
  std::string m_connectionString;
  std::string m_userName;
  std::string m_password;
};

#endif
