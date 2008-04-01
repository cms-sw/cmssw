#ifndef CondCore_DBCommon_SQLiteProxy_h
#define CondCore_DBCommon_SQLiteProxy_h
#include "CondCore/DBCommon/interface/TechnologyProxy.h"
namespace cond{
  class SQLiteProxy:public TechnologyProxy{
  public:
    SQLiteProxy();
    ~SQLiteProxy();
    std::string getRealConnectString( const std::string& iValue ) const;
    void setupSession();
    void prepareConnection();
    void prepareTransaction();
  };
}//ns cond
#endif
