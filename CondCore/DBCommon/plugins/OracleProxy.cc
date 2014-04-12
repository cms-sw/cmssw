#include "CondCore/DBCommon/interface/TechnologyProxy.h"
namespace cond {
  class OracleProxy: public TechnologyProxy {
  public:
    OracleProxy(){}
    ~OracleProxy(){}
    void initialize( const DbConnection& ) override{
    }
    std::string getRealConnectString( const std::string &userconnect ) const override{ return userconnect;}
    std::string getRealConnectString( const std::string &userconnect, const std::string& ) const override{ return userconnect;}
    bool isTransactional() const override { return true;}
  };
}//ns cond


#include "CondCore/DBCommon/interface/TechnologyProxyFactory.h"
DEFINE_EDM_PLUGIN(cond::TechnologyProxyFactory,cond::OracleProxy,"oracle");
