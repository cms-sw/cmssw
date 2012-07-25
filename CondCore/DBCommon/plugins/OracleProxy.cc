#include "CondCore/DBCommon/interface/TechnologyProxy.h"
namespace cond {
  class OracleProxy: public TechnologyProxy {
  public:
    OracleProxy(){}
    ~OracleProxy(){}
    void initialize(const std::string &userconnect, const DbConnection&){
      m_userconnect = userconnect;
    }
    std::string getRealConnectString() const{ return  m_userconnect;}
    std::string getRealConnectString( const std::string& ) const{ return m_userconnect;}
    bool isTransactional() const { return true;}
    std::string m_userconnect;
  };
}//ns cond


#include "CondCore/DBCommon/interface/TechnologyProxyFactory.h"
DEFINE_EDM_PLUGIN(cond::TechnologyProxyFactory,cond::OracleProxy,"oracle");
