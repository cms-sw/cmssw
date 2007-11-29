#ifndef CondCore_DBCommon_IConnectionProxy_H
#define CondCore_DBCommon_IConnectionProxy_H
#include <string>
#include <boost/timer.hpp>
//
// Package:     DBCommon
// Class  :     IConnectionProxy
// 
/**\class IConnectionProxy IConnectionProxy.h CondCore/DBCommon/interface/IConnectionProxy.h
   Description: developer interface for connection 
*/
//
// Author:      Zhen Xie
//
namespace coral{
  class IConnectionService;
}
namespace cond{
  class ITransaction;
  class IConnectionProxy{
  public:
    /// Constructor. A connection proxy holds connection parameters shared by underlying implementations
    IConnectionProxy(coral::IConnectionService* connectionServiceHandle, 
		     const std::string& con, 
		     int connectionTimeOut);
    /// Destructor
    virtual ~IConnectionProxy();
    /// real connection time out parameter
    virtual int connectionTimeOut() const;
    /// connection string 
    virtual std::string connectStr() const;
    /// return transaction handle. Child class must implement this method
    virtual ITransaction&  transaction() = 0;    
  protected:
    //all the child connection proxies must hold connection service handle
    coral::IConnectionService* m_connectionSvcHandle;
    std::string m_con;
    int m_connectionTimeOut;
    boost::timer m_timer;    
  };
}
#endif
