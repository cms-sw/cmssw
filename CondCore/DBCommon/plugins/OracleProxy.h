#ifndef CondCore_DBCommon_OracleProxy_h
#define CondCore_DBCommon_OracleProxy_h
#include "CondCore/DBCommon/interface/TechnologyProxy.h"
namespace cond{
  class OracleProxy:public TechnologyProxy{
  public:
    OracleProxy();
    ~OracleProxy();
    std::string getRealConnectString( const std::string& iValue ) const;
    void setupSession();
    void prepareConnection();
    void prepareTransaction();
  };
}//ns cond
#endif
