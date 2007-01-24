#ifndef CondCore_ESSources_PoolDBESSource_h
#define CondCore_ESSources_PoolDBESSource_h
// system include files
//#include <memory>
#include <string>
#include <map>
// user include files
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
namespace edm{
  class ParameterSet;
}
namespace cond{
  class DBSession;
  class PoolStorageManager;
  class IOVService;
}
class PoolDBESSource : public edm::eventsetup::DataProxyProvider,public edm::EventSetupRecordIntervalFinder{
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
  typedef std::multimap<std::string, std::string> RecordToTypes;
  RecordToTypes m_recordToTypes; //should be static?
  typedef std::string proxyName;
  typedef std::string tokenType;
  typedef std::map<proxyName, tokenType > ProxyToToken;
  ProxyToToken m_proxyToToken;
  typedef ProxyToToken::iterator pProxyToToken;
  std::string m_con;
  typedef std::map<std::string, std::string > RecordToIOV;
  RecordToIOV m_recordToIOV;
  //std::vector< std::pair < std::string, std::string> > m_recordToTag;
  std::string m_timetype;
  cond::DBSession* m_session;
  cond::IOVService* m_iovservice;
  cond::PoolStorageManager* m_pooldb;
  //bool m_tagTranslated;
  //std::string m_catalog; 
  bool m_connected;
 private:
  void tagToToken(const std::vector< std::pair < std::string, std::string> >& recordToTag);
  //void initIOV();
};
#endif
