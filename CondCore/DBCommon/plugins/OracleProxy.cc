#include "CondCore/DBCommon/interface/TechnologyProxy.h"
namespace cond {
  class OracleProxy: public TechnologyProxy {
  public:
    OracleProxy(){}
    ~OracleProxy(){}
    void initialize( const DbConnection& ){
    }
    std::string getRealConnectString( const std::string &userconnect ) const{ return userconnect;}
    std::string getRealConnectString( const std::string &userconnect, const std::string& ) const{ return userconnect;}
    bool isTransactional() const { return true;}
  };
}//ns cond


#include "CondCore/DBCommon/interface/TechnologyProxyFactory.h"
DEFINE_EDM_PLUGIN(cond::TechnologyProxyFactory,cond::OracleProxy,"oracle");
