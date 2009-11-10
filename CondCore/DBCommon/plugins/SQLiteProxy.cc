#include "SQLiteProxy.h"
#include "CondCore/DBCommon/interface/FipProtocolParser.h"
#include "CondCore/DBCommon/interface/TechnologyProxy.h"
#include <string>
namespace cond{
  class SQLiteProxy:public TechnologyProxy{
  public:
    explicit SQLiteProxy({}
    ~SQLiteProxy(){}
    void initialize(const DbConnection&){}
    std::string 
    getRealConnectString( ) const{
      std::string const & userconnect = m_session.connectionString();
      if( m_userconnect.find("sqlite_fip:") != std::string::npos ){
	cond::FipProtocolParser p;
	return p.getRealConnect(userconnect);
      }
      return userconnect;
    }
  };  
}//ns cond

#include "CondCore/DBCommon/interface/TechnologyProxyFactory.h"
DEFINE_EDM_PLUGIN(cond::TechnologyProxyFactory,cond::SQLiteProxy,"sqlite");
  
