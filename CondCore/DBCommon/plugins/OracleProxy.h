#ifndef CondCore_DBCommon_OracleProxy_h
#define CondCore_DBCommon_OracleProxy_h
#include "CondCore/DBCommon/interface/TechnologyProxy.h"
namespace cond{
  class DBSession;
  class OracleProxy:public TechnologyProxy{
  public:
    explicit OracleProxy( const std::string& userconnect );
    ~OracleProxy();
    std::string getRealConnectString() const;
    void setupSession( cond::DBSession& session );
  };
}//ns cond
#endif
