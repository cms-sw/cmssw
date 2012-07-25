#include "CondCore/DBCommon/interface/FipProtocolParser.h"
#include "CondCore/DBCommon/interface/TechnologyProxy.h"
#include <string>
namespace cond{
  class SQLiteProxy:public TechnologyProxy{
  public:
    SQLiteProxy(){}
    ~SQLiteProxy(){}
    void initialize(const std::string &userconnect, const DbConnection&){
        m_userconnect = userconnect;
    }
    std::string 
    getRealConnectString( ) const{
      if( m_userconnect.find("sqlite_fip:") != std::string::npos ){
	cond::FipProtocolParser p;
	return p.getRealConnect(m_userconnect);
      }
      return m_userconnect;
    }

    std::string 
    getRealConnectString( const std::string& ) const {
      return getRealConnectString();
    }

    bool isTransactional() const { return true;}

    std::string m_userconnect;
  };  
}//ns cond

#include "CondCore/DBCommon/interface/TechnologyProxyFactory.h"
DEFINE_EDM_PLUGIN(cond::TechnologyProxyFactory,cond::SQLiteProxy,"sqlite");
  
