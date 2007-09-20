#ifndef CondCore_ESSources_PoolDBESSource_h
#define CondCore_ESSources_PoolDBESSource_h
// system include files
//#include <memory>
#include <string>
#include <map>
// user include files
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "CondCore/DBCommon/interface/TagMetadata.h"
namespace edm{
  class ParameterSet;
}
namespace cond{
  class DBSession;
  class Connection;
  struct IOVInfo{
    std::string tag; 
    std::string token;
  };
}
class PoolDBESSource : public edm::eventsetup::DataProxyProvider,
		       public edm::EventSetupRecordIntervalFinder{
 public:
  PoolDBESSource( const edm::ParameterSet& );
  ~PoolDBESSource();
  
 protected:
  virtual void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
			      const edm::IOVSyncValue& , 
			      edm::ValidityInterval&) ;
  virtual void registerProxies(const edm::eventsetup::EventSetupRecordKey& iRecordKey, KeyedProxies& aProxyList) ;
  virtual void newInterval(const edm::eventsetup::EventSetupRecordKey& iRecordType, const edm::ValidityInterval& iInterval) ;    
 private:
  // ----------member data ---------------------------
  typedef std::multimap< std::string, std::string > RecordToTypes;
  RecordToTypes m_recordToTypes; 
  typedef std::map< std::string, cond::IOVInfo > RecordToIOVInfo;
  RecordToIOVInfo m_recordToIOVInfo;
  typedef std::map< std::string, cond::TagMetadata > TagCollection;
  TagCollection m_tagCollection;
  typedef std::map<std::string, std::string > ProxyToToken;
  ProxyToToken m_proxyToToken;
  cond::DBSession* m_session;
 private:
  void fillRecordToIOVInfo();
  std::string buildRecordTagKey( const std::string& recordName, 
				 const std::string& tagName );  
  std::string setupFrontier(const std::string& frontierconnect);
};
#endif
