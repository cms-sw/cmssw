#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbConnection.h"
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
    void initialize(const std::string&userconnect,const DbConnection& connection);
    std::string getRealConnectString() const;
    bool isTransactional() const { return false;}

  private:
    static unsigned int countslash(const std::string& input);
  private:
    std::string m_userconnect;
    std::string m_transactionId;
  std::vector<std::string> m_refreshtablelist;
  };
}//ns cond


cond::FrontierProxy::FrontierProxy(){
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

std::string 
cond::FrontierProxy::getRealConnectString() const{
  std::string result = m_userconnect;
  std::string proto("frontier://");
  std::string::size_type fpos=m_userconnect.find(proto);
  unsigned int nslash=this->countslash(m_userconnect.substr(proto.size(),m_userconnect.size()-fpos));
  if(nslash==1){
    edm::Service<edm::SiteLocalConfig> localconfservice;
    if( !localconfservice.isAvailable() ){
      throw cms::Exception("edm::SiteLocalConfigService is not available");       
    }
    result=localconfservice->lookupCalibConnect(m_userconnect);
  }
  if (!m_transactionId.empty()) {
    size_t l = result.rfind('/');
    result.insert(l,"(freshkey="+m_transactionId+')');
    
  }
  return result;
}

void 
cond::FrontierProxy::initialize(const std::string&userconnect, const DbConnection& connection) {
  m_userconnect = userconnect;
  m_transactionId = connection.configuration().transactionId();

  std::string refreshConnect;
  std::string realconnect=this->getRealConnectString();
  std::string::size_type startRefresh = realconnect.find("://");
  if (startRefresh != std::string::npos){
    startRefresh += 3;
  }
  std::string::size_type endRefresh=realconnect.rfind("/", std::string::npos);
  if (endRefresh == std::string::npos){
    refreshConnect = realconnect;
  }else{
    refreshConnect = realconnect.substr(startRefresh, endRefresh-startRefresh);
    if(refreshConnect.substr(0,1) != "("){
      //if the connect string is not a complicated parenthesized string,
      // an http:// needs to be at the beginning of it
      refreshConnect.insert(0, "http://");
    }
  }
  std::vector<std::string>::iterator ibeg=m_refreshtablelist.begin();
  std::vector<std::string>::iterator iend=m_refreshtablelist.end();
  for(std::vector<std::string>::iterator it=ibeg; it!=iend; ++it){
    connection.webCacheControl().refreshTable(refreshConnect,*it );
  }
  
}
unsigned int
cond::FrontierProxy::countslash(const std::string& input) {
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

#include "CondCore/DBCommon/interface/TechnologyProxyFactory.h"
DEFINE_EDM_PLUGIN(cond::TechnologyProxyFactory,cond::FrontierProxy,"frontier");
