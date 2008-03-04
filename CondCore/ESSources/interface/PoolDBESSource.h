#ifndef CondCore_ESSources_PoolDBESSource_h
#define CondCore_ESSources_PoolDBESSource_h
//
// Package:    CondCore/ESSources
// Class:      PoolDBESSource
// 
/**\class PoolDBESSource PoolDBESSource.h CondCore/ESSources/interface/PoolDBESSource.h
 Description: EventSetup source module for serving data from offline database
*/
//
// Author:      Zhen Xie
//

// system include files
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
  class PoolStorageManager;
  class IOVService;
  struct IOVInfo{
    std::string tag; 
    std::string token;
    std::string label;
    std::string pfn;
    std::string timetype;
  };
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
  typedef std::map<std::string,std::vector<cond::IOVInfo> > ProxyToIOVInfo;
  ProxyToIOVInfo m_proxyToIOVInfo;
  typedef std::map< std::string, cond::TagMetadata > TagCollection;
  TagCollection m_tagCollection;
  typedef std::map<std::string, std::string > DatumToToken;
  DatumToToken m_datumToToken;
  cond::DBSession* m_session;
  cond::IOVService* m_iovservice;
  cond::PoolStorageManager* m_pooldb;
  std::string m_timetype;
  bool m_connected; 
  std::string m_con;
 private:
  void fillRecordToIOVInfo();
  unsigned int countslash(const std::string& input)const;
};
#endif
