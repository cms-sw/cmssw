#include "CondCore/DBCommon/interface/FipProtocolParser.h"
#include "CondCore/DBCommon/interface/TechnologyProxy.h"
#include <string>
namespace cond{
  class SQLiteProxy:public TechnologyProxy{
  public:
    SQLiteProxy(){}
    ~SQLiteProxy(){}
    void initialize( const DbConnection& ) override{
    }
    std::string 
    getRealConnectString( const std::string &userconnect ) const override{
      if( userconnect.find("sqlite_fip:") != std::string::npos ){
	cond::FipProtocolParser p;
	return p.getRealConnect( userconnect );
      }
      return userconnect;
    }

    std::string 
    getRealConnectString( const std::string &userconnect, const std::string& ) const override {
      return getRealConnectString( userconnect );
    }

    bool isTransactional() const override { return true;}

  };  
}//ns cond

#include "CondCore/DBCommon/interface/TechnologyProxyFactory.h"
DEFINE_EDM_PLUGIN(cond::TechnologyProxyFactory,cond::SQLiteProxy,"sqlite");
  
