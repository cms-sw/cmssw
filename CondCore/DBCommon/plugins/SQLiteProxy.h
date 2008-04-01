#ifndef CondCore_DBCommon_SQLiteProxy_h
#define CondCore_DBCommon_SQLiteProxy_h
#include "CondCore/DBCommon/interface/TechnologyProxy.h"
#include <string>
namespace cond{
  class DBSession;
  class SQLiteProxy:public TechnologyProxy{
  public:
    explicit SQLiteProxy( const std::string& userconnect );
    ~SQLiteProxy();
    std::string getRealConnectString() const;
    void setupSession( cond::DBSession& session );
  };
}//ns cond
#endif
