// -*- C++ -*-
//
// Package:    PoolDBESSource
// Class:      PoolDBESSource
// 
/**\class PoolDBESSource PoolDBESSource.h CondCore/ESSources/interface/PoolDBESSource.h

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Chris Jones
//         Created:  Sat Jul 23 14:57:44 EDT 2005
// $Id: PoolDBESSource.cc,v 1.2 2005/08/29 17:36:33 xiezhen Exp $
//
//


// system include files
#include <memory>
#include "boost/shared_ptr.hpp"
#include <iostream>
#include <map>

// user include files
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "CondCore/PluginSystem/interface/ProxyFactory.h"
#include "CondCore/IOVService/interface/IOV.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "PluginManager/PluginManager.h"
#include "POOLCore/Token.h"
#include "FileCatalog/URIParser.h"
#include "FileCatalog/FCSystemTools.h"
#include "FileCatalog/IFileCatalog.h"
#include "StorageSvc/DbType.h"
#include "PersistencySvc/DatabaseConnectionPolicy.h"
#include "PersistencySvc/ISession.h"
#include "PersistencySvc/ITransaction.h"
#include "PersistencySvc/IDatabase.h"
#include "PersistencySvc/Placement.h"
#include "DataSvc/DataSvcFactory.h"
#include "DataSvc/IDataSvc.h"
#include "DataSvc/ICacheSvc.h"
#include "DataSvc/Ref.h"
#include "POOLCore/POOLContext.h"
#include "SealKernel/Exception.h"

#include "RelationalAccess/RelationalException.h"
//
// class decleration
//

class PoolDBESSource : public edm::eventsetup::DataProxyProvider, 
		       public edm::eventsetup::EventSetupRecordIntervalFinder
{
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
  std::string m_tag;
  std::string m_timetype;
  std::auto_ptr<pool::IFileCatalog> m_cat;
  pool::IDataSvc* m_svc;
  pool::Ref<cond::IOV> m_iov;
private:
  void initPool();
  void closePool();
  bool initIOV();
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//
static
std::string
buildName( const std::string& iRecordName, const std::string& iTypeName ) {
  //std::cout<<"building name "<<iRecordName+"_"+iTypeName+"_Proxy"<<std::endl;
  return iRecordName+"_"+iTypeName+"_Proxy";
}

void PoolDBESSource::initPool(){
  try{
    pool::POOLContext::loadComponent( "SEAL/Services/MessageService" );
    pool::POOLContext::setMessageVerbosityLevel( seal::Msg::Error );
    // needed to connect to oracle
    pool::POOLContext::loadComponent( "POOL/Services/EnvironmentAuthenticationService" );
    pool::URIParser p;
    p.parse();    
    // the required lifetime of the file catalog is the same of the  srv_ or longer  
    m_cat.reset(new pool::IFileCatalog);
    m_cat->addReadCatalog(p.contactstring());
    m_cat->connect();
    m_cat->start();    
    m_svc= pool::DataSvcFactory::instance(&(*m_cat));
    // Define the policy for the implicit file handling
    pool::DatabaseConnectionPolicy policy;
    policy.setWriteModeForNonExisting(pool::DatabaseConnectionPolicy::CREATE);
    m_svc->session().setDefaultConnectionPolicy(policy);
    m_svc->transaction().start(pool::ITransaction::READ);
  }catch(seal::Exception& e){
    std::cerr << e.what() << std::endl;
  } catch ( std::exception& e ) {
    std::cerr << e.what() << std::endl;
  } catch ( ... ) {
    std::cerr << "Funny error" << std::endl;
  }
}

void PoolDBESSource::closePool(){
  m_svc->transaction().commit();
  m_svc->session().disconnectAll();
  m_cat->commit();
  m_cat->disconnect();
  if(m_svc) delete m_svc;
}

bool PoolDBESSource::initIOV(){
  cond::MetaData meta(m_con);
  std::string iovToken("");
  try{
    iovToken=meta.getToken(m_tag);
    if( iovToken.empty() ){
      return false;
    }
  }catch( const pool::RelationalTableNotFound& e ){
    std::cerr<<e.what()<<std::endl;
    return false;
  }catch(const seal::Exception&e ){
    std::cerr<<e.what()<<std::endl;
    return false;
  }
  pool::Ref<cond::IOV> iov(m_svc, iovToken);
  std::pair<int,std::string> iovpair=*iov->iov.lower_bound(7);
  m_iov=iov;
  return true;
}

//
// constructors and destructor
//
PoolDBESSource::PoolDBESSource( const edm::ParameterSet& iConfig ) :
  m_con(iConfig.getParameter<std::string>("connect") ),
  m_tag(iConfig.getParameter<std::string>("tag") ),
  m_timetype(iConfig.getParameter<std::string>("timetype") )
{
  using namespace std;
  using namespace edm;
  using namespace edm::eventsetup;
  if( iConfig.getParameter<bool>("loadAll") ) {
    m_recordToTypes.insert(make_pair(string("EcalPedestalsRcd"), string("EcalPedestals"))) ;
    //m_recordToTypes.insert(make_pair(string("HcalPedestalsRcd"), string("HcalPedestals"))) ;
    //by forcing this to load, we also load the definition of the Records which //will allow EventSetupRecordKey::TypeTag::findType(...) method to find them
    for(RecordToTypes::iterator itRec = m_recordToTypes.begin();itRec != m_recordToTypes.end();	++itRec ) {
      m_proxyToToken.insert( make_pair(buildName(itRec->first, itRec->second ),"") );//fill in dummy tokens now, change in setIntervalFor
      pProxyToToken pos=m_proxyToToken.find(buildName(itRec->first, itRec->second));
      //m_svc->transaction().start(pool::ITransaction::READ);
      boost::shared_ptr<DataProxy> proxy(cond::ProxyFactory::get()->create( buildName(itRec->first, itRec->second),m_svc,pos));
      //m_svc->transaction().commit();
    }
  }else{
    std::cerr<<"what else?"<<std::endl;
    throw cms::Exception("Unsupported operation.");
  }
  
  //NOTE: should delay setting what  records until all 
  string lastRecordName;
  for(RecordToTypes::const_iterator itName = m_recordToTypes.begin();itName != m_recordToTypes.end();++itName ) {
    if( lastRecordName != itName->first ) {
      lastRecordName = itName->first;
      //std::cout<<"lastRecordName "<<lastRecordName<<std::endl;
      EventSetupRecordKey recordKey = EventSetupRecordKey::TypeTag::findType( itName->first );
      if ( recordKey == EventSetupRecordKey() ) {
	cout << "The Record type named \""<<itName->first<<"\" could not be found.  We therefore assume it is not needed for this job"
	     << endl;
      }
      findingRecordWithKey( recordKey );
      usingRecordWithKey( recordKey );
    }
  }
  ///
  //now do what ever other initialization is needed
  ///
  this->initPool();
  if( !this->initIOV() ){
    throw cms::Exception("IOV not found for "+m_tag);
  }
}


PoolDBESSource::~PoolDBESSource()
{
  this->closePool();
}


//
// member functions
//
void 
PoolDBESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& iKey, const edm::IOVSyncValue& iTime, edm::ValidityInterval& oInterval ) {
  //std::cout<<" PoolDBESSource::setIntervalFor"<<std::endl;
  RecordToTypes::iterator itRec= m_recordToTypes.find( iKey.name() );
  if( itRec == m_recordToTypes.end() ) {
    //no valid record
    oInterval = edm::ValidityInterval::invalidInterval();
    return;
  }
  std::string payloadToken;
  typedef std::map<int, std::string> IOVMap;
  typedef IOVMap::const_iterator iterator;
  unsigned long abtime=iTime.eventID().run()-edm::IOVSyncValue::beginOfTime().eventID().run();
  iterator iEnd = m_iov->iov.lower_bound( abtime );
  if( iEnd == m_iov->iov.end() ||  (*iEnd).second.empty() ) {
    //no valid data
    oInterval = edm::ValidityInterval(edm::IOVSyncValue::endOfTime(), edm::IOVSyncValue::endOfTime());
  } else {
    unsigned long starttime=edm::IOVSyncValue::beginOfTime().eventID().run();
    if (iEnd != m_iov->iov.begin()) {
      iterator iStart(iEnd); iStart--;
      starttime = (*iStart).first+edm::IOVSyncValue::beginOfTime().eventID().run();
    }
    payloadToken = (*iEnd).second;
    //std::cout<<"payloadToken "<<payloadToken<<std::endl;
    edm::IOVSyncValue start( edm::EventID(starttime,0) );
    edm::IOVSyncValue stop ( edm::EventID((*iEnd).first+edm::IOVSyncValue::beginOfTime().eventID().run(),0) );
    oInterval = edm::ValidityInterval( start, stop );
  }
  m_proxyToToken[buildName(itRec->first ,itRec->second)]=payloadToken;  
}   

void 
PoolDBESSource::registerProxies(const edm::eventsetup::EventSetupRecordKey& iRecordKey , KeyedProxies& aProxyList) 
{
   using namespace edm;
   using namespace edm::eventsetup;
   using namespace std;
   cout <<string("registering Proxies for ") + iRecordKey.name() << endl;
   //For each data type in this Record, create the proxy by dynamically loading it
   std::pair< RecordToTypes::iterator,RecordToTypes::iterator > typeItrs = m_recordToTypes.equal_range( iRecordKey.name() );
   //loop over types in the same record
   for( RecordToTypes::iterator itType = typeItrs.first; itType != typeItrs.second; ++itType ) {
     cout <<string("   ") + itType->second ;
     static eventsetup::TypeTag defaultType;
     eventsetup::TypeTag type = eventsetup::TypeTag::findType( itType->second );
     //std::cout<<"default type "<<std::string(defaultType.name())<<std::endl;
     if( type != defaultType ) {
       pProxyToToken pos=m_proxyToToken.find(buildName(iRecordKey.name(), type.name()));
       //m_svc->transaction().start(pool::ITransaction::READ);
       boost::shared_ptr<DataProxy> proxy(cond::ProxyFactory::get()->create( buildName(iRecordKey.name(), type.name() ), m_svc, pos));
       //m_svc->transaction().commit();
       cout <<string("   ") + type.name() ;
       if(0 != proxy.get()) {
	 eventsetup::DataKey key( type, "");
	 aProxyList.push_back(KeyedProxies::value_type(key,proxy));
       }
     }
     cout <<endl;
   }
}


void 
PoolDBESSource::newInterval(const edm::eventsetup::EventSetupRecordKey& iRecordType,const edm::ValidityInterval& iInterval) 
{
  invalidateProxies(iRecordType);
}

// ------------ method called to produce the data  ------------

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_SOURCE(PoolDBESSource)
  
