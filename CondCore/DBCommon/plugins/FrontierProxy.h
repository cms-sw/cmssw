#ifndef CondCore_DBCommon_FrontierProxy_h
#define CondCore_DBCommon_FrontierProxy_h
#include "CondCore/DBCommon/interface/TechnologyProxy.h"
#include <string>
#include <vector>
namespace cond{
  class DBSession;
  class FrontierProxy:public TechnologyProxy{
  public:
    explicit FrontierProxy( const std::string& userconnect );
    ~FrontierProxy();
    std::string getRealConnectString() const;
    void setupSession( cond::DBSession& session );
  private:
    std::vector<std::string> m_refreshtablelist;
  private:
    unsigned int countslash(const std::string& input)const;
  };
}//ns cond
#endif
