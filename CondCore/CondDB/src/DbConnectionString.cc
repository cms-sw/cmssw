#include "CondCore/CondDB/interface/Exception.h"
#include "CondCore/CondDB/interface/Utils.h"
#include "DbConnectionString.h"
//
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

namespace cond {

  namespace persistency {

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

    std::pair<std::string,std::string> makeFrontierConnectionString( const std::string& initialConnection, 
					      const std::string& transactionId ){
      std::string realConn = initialConnection;
      std::string proto("frontier://");
      std::string::size_type fpos=initialConnection.find(proto);
      unsigned int nslash=countslash(initialConnection.substr(proto.size(),initialConnection.size()-fpos));
      if(nslash==1){
	edm::Service<edm::SiteLocalConfig> localconfservice;
	if( !localconfservice.isAvailable() ){
	  throwException("SiteLocalConfigService is not available","cond::makeRealConnectString");       
	}
	realConn=localconfservice->lookupCalibConnect(initialConnection);
      }
      if (!transactionId.empty()) {
	size_t l = realConn.rfind('/');
	realConn.insert(l,"(freshkey="+transactionId+')');
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
      return std::make_pair(realConn,refreshConnect);
    }
    
    std::pair<std::string,std::string> getRealConnectionString( const std::string& initialConnection ){
      return getRealConnectionString( initialConnection, "" );
    }
    
    std::pair<std::string,std::string> getRealConnectionString( const std::string& initialConnection, 
								const std::string& transId ){
      auto connData = parseConnectionString( initialConnection );
      if( std::get<0>(connData) == "frontier" ) return makeFrontierConnectionString(  initialConnection, transId );    
      return std::make_pair(initialConnection,"");
    }

  }
}
