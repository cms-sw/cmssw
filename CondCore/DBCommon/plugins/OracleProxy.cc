#include "CondCore/DBCommon/interface/TechnologyProxy.h"
namespace cond {
  class OracleProxy: public TechnologyProxy {
  public:
    explicit OracleProxy(){}
    ~OracleProxy(){}
    void initialize(const DbConnection&){}
    std::string getRealConnectString() const{ return m_session.connectionString();}
  };
}//ns cond


#include "CondCore/DBCommon/interface/TechnologyProxyFactory.h"
DEFINE_EDM_PLUGIN(cond::TechnologyProxyFactory,cond::OracleProxy,"oracle");
