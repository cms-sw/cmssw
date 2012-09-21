#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "RelationalAccess/IWebCacheControl.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//#include "CondCore/MetaDataService/interface/MetaDataNames.h"
#include "CondCore/IOVService/interface/IOVNames.h"

#include "CondCore/DBCommon/interface/TechnologyProxy.h"
#include <string>
#include <vector>

namespace cond{
  class FrontierProxy: public TechnologyProxy{
  public:
    FrontierProxy();
    ~FrontierProxy();
    void initialize( const DbConnection& connection);
    std::string getRealConnectString( const std::string&userconnect ) const;
    std::string getRealConnectString( const std::string&userconnect, const std::string& transactionId ) const;
    bool isTransactional() const { return false;}

  private:
    //static unsigned int countslash(const std::string& input);
    std::string makeRealConnectString( const std::string& initialConnection, const std::string& transactionId ) const ;
  private:
    const DbConnection* m_dbConnection;
    std::vector<std::string> m_refreshtablelist;
  };
}//ns cond


cond::FrontierProxy::FrontierProxy():
  m_dbConnection(0),
  m_refreshtablelist(){
  m_refreshtablelist.reserve(10);
  //table names for IOVSequence in the old POOL mapping
  m_refreshtablelist.push_back(cond::IOVNames::iovTableName());
  m_refreshtablelist.push_back(cond::IOVNames::iovDataTableName());
  //table names for IOVSequence in ORA
  //FIXME: do not use hard-coded names, ORA should provide them for a given container...
  m_refreshtablelist.push_back("ORA_C_COND_IOVSEQUENCE");
  m_refreshtablelist.push_back("ORA_C_COND_IOVSEQU_A0");
  m_refreshtablelist.push_back("ORA_C_COND_IOVSEQU_A1");
  // do not refesh tag table in production...
  // m_refreshtablelist.push_back(cond::MetaDataNames::metadataTable());
}

cond::FrontierProxy::~FrontierProxy(){
  m_refreshtablelist.clear();
}

namespace cond {

  unsigned int
  countslash(const std::string& input) {
    unsigned int count=0;
    std::string::size_type slashpos( 0 );
    while( slashpos!=std::string::npos){
      slashpos = input.find('/', slashpos );
      if ( slashpos != std::string::npos ){
	++count;
	// start next search after this word
	slashpos += 1;
      }
    }
    return count;
  }

  std::string 
  cond::FrontierProxy::makeRealConnectString( const std::string& initialConnection, 
					      const std::string& transactionId ) const {
    std::string realConn = initialConnection;
    // for testing
    //std::string res = initialConnection;
    std::string proto("frontier://");
    std::string::size_type fpos=initialConnection.find(proto);
    unsigned int nslash=countslash(initialConnection.substr(proto.size(),initialConnection.size()-fpos));
    if(nslash==1){
      edm::Service<edm::SiteLocalConfig> localconfservice;
      if( !localconfservice.isAvailable() ){
	throw cms::Exception("edm::SiteLocalConfigService is not available");       
      }
      realConn=localconfservice->lookupCalibConnect(initialConnection);
      //res=localconfservice->lookupCalibConnect(initialConnection);
    }
    if (!transactionId.empty()) {
      size_t l = realConn.rfind('/');
      realConn.insert(l,"(freshkey="+transactionId+')');
      //size_t l = res.rfind('/');
      //res.insert(l,"(freshkey="+transactionId+')');
    }

    std::string refreshConnect;
    std::string::size_type startRefresh = realConn.find("://");
    if (startRefresh != std::string::npos){
      startRefresh += 3;
    }
    std::string::size_type endRefresh=realConn.rfind("/", std::string::npos);
    if (endRefresh == std::string::npos){
      refreshConnect = realConn;
    }else{
      refreshConnect = realConn.substr(startRefresh, endRefresh-startRefresh);
      if(refreshConnect.substr(0,1) != "("){
	//if the connect string is not a complicated parenthesized string,
	// an http:// needs to be at the beginning of it
	refreshConnect.insert(0, "http://");
      }
    }
    std::vector<std::string>::const_iterator ibeg=m_refreshtablelist.begin();
    std::vector<std::string>::const_iterator iend=m_refreshtablelist.end();
    for(std::vector<std::string>::const_iterator it=ibeg; it!=iend; ++it){
      m_dbConnection->webCacheControl().refreshTable(refreshConnect,*it );
    }

    //std::cout << "***** frontier connection string " << std::endl;
    //std::cout << res << std::endl;
    return realConn;
  }

}

std::string 
cond::FrontierProxy::getRealConnectString( const std::string&userconnect ) const {
  if( ! m_dbConnection ){
    throwException("The Technology Proxy has not been initialized.","FrontierProxy::getRealConnectString");
  }
  return makeRealConnectString(  userconnect, m_dbConnection->configuration().transactionId() );
}

std::string 
cond::FrontierProxy::getRealConnectString( const std::string&userconnect, const std::string& transId ) const{
  if( ! m_dbConnection ){
    throwException("The Technology Proxy has not been initialized.","FrontierProxy::getRealConnectString");
  }
  const std::string* tId = &transId;
  if( transId.empty() ) tId = &m_dbConnection->configuration().transactionId();
  return makeRealConnectString(  userconnect, *tId );
}

void 
cond::FrontierProxy::initialize( const DbConnection& connection ) {
  m_dbConnection = &connection;  
}

#include "CondCore/DBCommon/interface/TechnologyProxyFactory.h"
DEFINE_EDM_PLUGIN(cond::TechnologyProxyFactory,cond::FrontierProxy,"frontier");
