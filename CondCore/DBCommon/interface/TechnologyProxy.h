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

#include <string>
namespace cond{
  class DBSession;
  class TechnologyProxy{
  public:
    explicit TechnologyProxy( const std::string& userconnect ):m_userconnect(userconnect){}
    virtual ~TechnologyProxy(){}
    virtual std::string getRealConnectString() const=0;
    virtual void setupSession( DBSession& session )=0;
  protected:
    std::string m_userconnect;
  private:
    TechnologyProxy( const TechnologyProxy& );
    const TechnologyProxy& operator=(const TechnologyProxy&); 
  };
}//ns cond
#endif
