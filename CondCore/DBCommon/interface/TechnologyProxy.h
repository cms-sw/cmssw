#ifndef CondCore_DBCommon_TechnologyProxy_h
#define CondCore_DBCommon_TechnologyProxy_h
//
// Package:     DBCommon
// Class  :     TechnologyProxy
//
/**\class  TechnologyProxy TechnologyProxy.h CondCore/DBCommon/interface/TechnologyProxy.h
   Description: Abstract interface for technology specific operations. The concrete instance implementing the interface is created by TechnologyProxyFactory and loaded by the plugin manager
*/
//
// Author:      Zhen Xie
//

#include "CondCore/DBCommon/interface/DbSession.h"
#include <string>
namespace cond{
  class DbSession;
  class TechnologyProxy{
  public:
    explicit TechnologyProxy( ){}
    virtual ~TechnologyProxy(){}
    virtual void initialize(const std::string&userconnect, const DbConnection& connection)=0;
    virtual std::string getRealConnectString() const=0;
    virtual std::string getRealConnectString( const std::string& transactionId ) const=0;
    virtual bool isTransactional() const=0;
  private:
    TechnologyProxy( const TechnologyProxy& );
    const TechnologyProxy& operator=(const TechnologyProxy&); 
  };
}//ns cond
#endif
