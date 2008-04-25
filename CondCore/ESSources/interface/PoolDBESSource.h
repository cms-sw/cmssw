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
#include <set>
// user include files
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "CondCore/DBCommon/interface/TagMetadata.h"
#include "CondCore/DBCommon/interface/Time.h"
#include <boost/functional/hash.hpp>
namespace edm{
  class ParameterSet;
}
namespace cond{
  class DBSession;
  class CoralTransaction;
  class Connection;
  struct IOVInfo{
    std::string tag; 
    std::string token;
    std::string label;
    std::string pfn;
    cond::TimeType timetype;
    std::size_t hashvalue()const{
      boost::hash<std::string> hasher;
      std::size_t result=hasher(token+label+pfn);
      return result;
    }
    bool  operator == (const IOVInfo& toCompare ) const {
      if(this->hashvalue()==toCompare.hashvalue()&&this->timetype==toCompare.timetype) return true;
     return false;
    }
    bool operator != (const IOVInfo& toCompare ) const {
      return !(*this==toCompare);
    }
    bool operator<(const IOVInfo& toCompare ) const {
      return this->hashvalue()<toCompare.hashvalue();
    }
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
  typedef std::map< std::string, std::set<cond::IOVInfo> > ProxyToIOVInfo;
  ProxyToIOVInfo m_proxyToIOVInfo;
  typedef std::map< std::string, cond::TagMetadata > TagCollection;
  TagCollection m_tagCollection;
  typedef std::map<std::string, std::string > DatumToToken;
  DatumToToken m_datumToToken;
  cond::DBSession* m_session;
 private:
  void fillRecordToIOVInfo();
  void fillTagCollectionFromDB( cond::CoralTransaction& coraldb,
				const std::string& roottag );
  //std::string setupFrontier(const std::string& frontierconnect);
  //unsigned int countslash(const std::string& input)const;
};
#endif
