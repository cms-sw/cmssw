#ifndef CondCore_DBCommon_FrontierProxy_h
#define CondCore_DBCommon_FrontierProxy_h
#include "CondCore/DBCommon/interface/TechnologyProxy.h"
namespace cond{
  class FrontierProxy:public TechnologyProxy{
  public:
    FrontierProxy();
    ~FrontierProxy();
    std::string getRealConnectString( const std::string& iValue ) const;
    void setupSession();
    void prepareConnection();
    void prepareTransaction();
  };
}//ns cond
#endif
