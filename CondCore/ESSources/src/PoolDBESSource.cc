// system include files
#include "boost/shared_ptr.hpp"
#include "CondCore/ESSources/interface/PoolDBESSource.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/DBCommon/interface/ConfigSessionFromParameterSet.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/FipProtocolParser.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/DataProxy.h"
#include "CondCore/PluginSystem/interface/ProxyFactory.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/MetaDataService/interface/MetaDataNames.h"
#include "POOLCore/Exception.h"
#include "RelationalAccess/IConnectionService.h"
#include "RelationalAccess/IWebCacheControl.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <exception>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FileCatalog/IFileCatalog.h"
#include <sstream>
#include <cstdlib>
//#include <iostream>
//
// static data member definitions
//

static
std::string
buildName( const std::string& iRecordName, const std::string& iTypeName ) {
  return iRecordName+"@"+iTypeName+"@Proxy";
}

static
std::pair<std::string,std::string>
deconstructName(const std::string& iProxyName) {
  if(iProxyName.find_first_of("@")==std::string::npos){
    return std::make_pair("","");
  }
  std::string recordName(iProxyName, 0, iProxyName.find_first_of("@"));
  std::string typeName(iProxyName,recordName.size()+1,iProxyName.size()-6-recordName.size()-1);
  //std::cout <<"Record \""<<recordName<<"\" type \""<<typeName<<"\""<<std::endl;
  return std::make_pair(recordName,typeName);
}

#include "FWCore/PluginManager/interface/PluginManager.h"

static
void
fillRecordToTypeMap(std::multimap<std::string, std::string>& oToFill){
  //From the plugin manager get the list of our plugins
  // then from the plugin names, we can deduce the 'record to type' information
  //std::cout<<"Entering fillRecordToTypeMap "<<std::endl;   
   edmplugin::PluginManager*db =  edmplugin::PluginManager::get();
   typedef edmplugin::PluginManager::CategoryToInfos CatToInfos;
   
   const std::string mycat(cond::pluginCategory());
   CatToInfos::const_iterator itFound = db->categoryToInfos().find(mycat);
   if(itFound == db->categoryToInfos().end()) {
      return;
   }
   std::string lastClass;
   for (edmplugin::PluginManager::Infos::const_iterator itInfo = itFound->second.begin(),
	   itInfoEnd = itFound->second.end(); 
	itInfo != itInfoEnd; ++itInfo)
   {
      if (lastClass == itInfo->name_) {
         continue;
      }
      
      lastClass = itInfo->name_;
      std::pair<std::string,std::string> pairName=deconstructName(lastClass);
      if( pairName.first.empty() ) continue;
      if( oToFill.find(pairName.first)==oToFill.end() ){
	 //oToFill.insert(deconstructName(cap));
	 oToFill.insert(pairName);
      }else{
	 for(std::multimap<std::string, std::string>::iterator pos=oToFill.lower_bound(pairName.first); pos != oToFill.upper_bound(pairName.first); ++pos ){
	    if(pos->second != pairName.second){
	       oToFill.insert(pairName);
	    }else{
	       //std::cout<<"ignore "<<pairName.first<<" "<<pairName.second<<std::endl;
	    }
	 }
      }
   }
}
//
// constructors and destructor
//
static cond::ConnectionHandler& conHandler=cond::ConnectionHandler::Instance();
PoolDBESSource::PoolDBESSource( const edm::ParameterSet& iConfig ) :
  m_timetype(iConfig.getParameter<std::string>("timetype") ),
  m_session( 0 )
{		
  //std::cout<<"PoolDBESSource::PoolDBESSource"<<std::endl;
  /*parameter set parsing and pool environment setting
   */
  std::string catStr;
  std::string connect;
  connect=iConfig.getParameter<std::string>("connect"); 
  if( connect.find("sqlite_fip:") != std::string::npos ){
    cond::FipProtocolParser p;
    connect=p.getRealConnect(connect);
  }
  m_session=new cond::DBSession;
  edm::ParameterSet connectionPset = iConfig.getParameter<edm::ParameterSet>("DBParameters"); 
  cond::ConfigSessionFromParameterSet configConnection(*m_session,connectionPset);
  //std::cout<<"using connect "<<connect<<std::endl;
  m_session->open();
  ///handle frontier cache refresh
  if( connect.rfind("frontier://") != std::string::npos){
    //Mark tables that need to not be cached (always refreshed)
    //strip off the leading protocol:// and trailing schema name from connect
    edm::Service<edm::SiteLocalConfig> localconfservice;
    if( !localconfservice.isAvailable() ){
      throw cms::Exception("edm::SiteLocalConfigService is not available");       
    }
    connect=localconfservice->lookupCalibConnect(connect);
    std::string::size_type startRefresh = connect.find("://");
    if (startRefresh != std::string::npos){
      startRefresh += 3;
    }
    std::string::size_type endRefresh = connect.rfind("/", std::string::npos);
    std::string refreshConnect;
    if (endRefresh == std::string::npos){
      refreshConnect = connect;
    }else{
      refreshConnect = connect.substr(startRefresh, endRefresh-startRefresh);
      if(refreshConnect.substr(0,1) != "("){
	//if the connect string is not a complicated parenthesized string,
	// an http:// needs to be at the beginning of it
	refreshConnect.insert(0, "http://");
      }
    }
    //get handle to webCacheControl()
    m_session->webCacheControl().refreshTable( refreshConnect,cond::IOVNames::iovTableName() );
    m_session->webCacheControl().refreshTable( refreshConnect,cond::IOVNames::iovDataTableName() );
    m_session->webCacheControl().refreshTable( refreshConnect,cond::MetaDataNames::metadataTable() );
  }
  conHandler.registerConnection("inputdb",connect,0);
  conHandler.connect(m_session);
  using namespace edm;
  using namespace edm::eventsetup;  
  fillRecordToTypeMap(m_recordToTypes);
  //parsing record to tag
  std::vector< std::pair<std::string,std::string> > recordToTag;
  typedef std::vector< ParameterSet > Parameters;
  Parameters toGet = iConfig.getParameter<Parameters>("toGet");
  if( 0==toGet.size() ){
    throw cond::Exception("Configuration") <<" The \"toGet\" parameter is empty, please specify what (Record, tag) pairs you wish to retrieve\n"
					   <<" or use the record name \"all\" to have your tag be used to retrieve all known Records\n";
  }
  std::string tagName;
  std::string recordName;
  std::string typeName;
  std::vector< std::pair<std::string, cond::TagMetadata> > tagcollection;
  std::string lastRecordName;
  for(Parameters::iterator itToGet = toGet.begin(); itToGet != toGet.end(); ++itToGet ) {
    cond::TagMetadata m;
    m.pfn="inputdb";
    m.recordname = itToGet->getParameter<std::string>("record");
    tagName = itToGet->getParameter<std::string>("tag");
    
    //load proxy code now to force in the Record code
    //std::cout<<"recordName "<<recordName<<std::endl;
    //std::cout<<"tagName "<<tagName<<std::endl;
    std::multimap<std::string, std::string>::iterator itFound=m_recordToTypes.find(m.recordname);
    if(itFound == m_recordToTypes.end()){
      throw cond::Exception("NoRecord")<<" The record \""<<m.recordname<<"\" is not known by the PoolDBESSource";
    }
    m.objectname=itFound->second;
    std::string proxyName = buildName(m.recordname,m.objectname);
    //std::cout<<"typeName "<<typeName<<std::endl;
    //std::cout<<"proxyName "<<proxyName<<std::endl;
    m_proxyToToken.insert( make_pair(proxyName,"") );
    //fill in dummy tokens now, change in setIntervalFor
    pProxyToToken pos=m_proxyToToken.find(proxyName);
    cond::Connection* c=conHandler.getConnection(m.pfn);
    boost::shared_ptr<DataProxy> proxy(cond::ProxyFactory::get()->create(proxyName,c,pos));
    eventsetup::EventSetupRecordKey recordKey(eventsetup::EventSetupRecordKey::TypeTag::findType( m.recordname ) );
    if( recordKey.type() == eventsetup::EventSetupRecordKey::TypeTag() ) {
      //record not found
      throw cond::Exception("NoRecord")<<"The record \""<< m.recordname <<"\" does not exist ";
    }
    recordToTag.push_back(std::make_pair(m.recordname, tagName));
    if( lastRecordName != m.recordname ) {
      lastRecordName = m.recordname;
      findingRecordWithKey( recordKey );
      usingRecordWithKey( recordKey );
    }
    tagcollection.push_back(std::make_pair<std::string,cond::TagMetadata>(tagName,m));
  }
  m_con=connect;
  this->tagToToken(tagcollection);
}
PoolDBESSource::~PoolDBESSource()
{
  delete m_session;
}
//
// member functions
//
void 
PoolDBESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& iKey, const edm::IOVSyncValue& iTime, edm::ValidityInterval& oInterval ) {
  LogDebug ("PoolDBESSource")<<iKey.name();
  //std::cout<<"PoolDBESSource::setIntervalFor "<< iKey.name() <<" at time "<<iTime.eventID().run()<<std::endl;
  RecordToTypes::iterator itRec = m_recordToTypes.find( iKey.name() );
  if( itRec == m_recordToTypes.end() ) {
    //no valid record
    LogDebug ("PoolDBESSource")<<"no valid record ";
    //std::cout<<"no valid record "<<std::endl;
    oInterval = edm::ValidityInterval::invalidInterval();
    return;
  }
  //std::cout<<"recordToIOV size "<<m_recordToIOV.size()<<std::endl;
  RecordToIOV::iterator itIOV = m_recordToIOV.find( iKey.name() );
  if( itIOV == m_recordToIOV.end() ){
    LogDebug ("PoolDBESSource")<<"no valid IOV found for record "<<iKey.name();
    oInterval = edm::ValidityInterval::invalidInterval();
    return;
  }
  std::string iovToken=itIOV->second;
  std::string payloadToken;
  cond::Time_t abtime;
  std::ostringstream os;
  if( m_timetype == "timestamp" ){
    abtime=(cond::Time_t)iTime.time().value();
  }else{
    abtime=(cond::Time_t)iTime.eventID().run();
  }
  //valid time check
  //check if current run exceeds iov upperbound
  /*standalone    
   */
  cond::Connection* c=conHandler.getConnection("inputdb");
  cond::PoolTransaction& pooldb=c->poolTransaction(true);
  cond::IOVService iovservice(pooldb);  
  pooldb.start();
  if( !iovservice.isValid(iovToken,abtime) ){
    os<<abtime;
    throw cond::noDataForRequiredTimeException("PoolDBESSource::setIntervalFor",iKey.name(),os.str());
  }
  std::pair<cond::Time_t, cond::Time_t> validity=iovservice.validity(iovToken,abtime);
  payloadToken=iovservice.payloadToken(iovToken,abtime);
  pooldb.commit();
  edm::IOVSyncValue start;
  if( m_timetype == "timestamp" ){
    start=edm::IOVSyncValue( edm::Timestamp(validity.first) );
  }else{
    start=edm::IOVSyncValue( edm::EventID(validity.first,0) );
  }
  edm::IOVSyncValue stop;
  if( m_timetype == "timestamp" ){
    stop=edm::IOVSyncValue( edm::Timestamp(validity.second) );
    LogDebug ("PoolDBESSource")
      <<" set start time "<<start.time().value()
      <<" ; set stop time "<<stop.time().value();
  }else{
    stop=edm::IOVSyncValue( edm::EventID(validity.second,0) );
    LogDebug ("PoolDBESSource")
      <<" set start run "<<start.eventID().run()
      <<" ; set stop run "<<stop.eventID().run();
  }
  oInterval = edm::ValidityInterval( start, stop );
  //std::cout<<"setting itRec->first "<<itRec->first<<std::endl;
  //std::cout<<"setting itRec->second "<<itRec->second<<std::endl;
  //std::cout<<"payloadToken "<< payloadToken<<std::endl;
  //std::cout<<"buildProxy "<<buildName(itRec->first ,itRec->second)<<std::endl;
  m_proxyToToken[buildName(itRec->first ,itRec->second)]=payloadToken;  
}   

void 
PoolDBESSource::registerProxies(const edm::eventsetup::EventSetupRecordKey& iRecordKey , KeyedProxies& aProxyList) 
{
  //LogDebug ("PoolDBESSource ")<<"registerProxies";
  using namespace edm;
  using namespace edm::eventsetup;
  //using namespace std;
  //std::cout <<"registering Proxies for "<< iRecordKey.name() << std::endl;
  //For each data type in this Record, create the proxy by dynamically loading it
  std::pair< RecordToTypes::iterator,RecordToTypes::iterator > typeItrs = m_recordToTypes.equal_range( iRecordKey.name() );
  //loop over types in the same record
  for( RecordToTypes::iterator itType = typeItrs.first; itType != typeItrs.second; ++itType ) {
    //std::cout<<"Entering loop PoolDBESSource::registerProxies"<<std::endl;
    //std::cout<<std::string("   ") + itType->second <<std::endl;
    static eventsetup::TypeTag defaultType;
    eventsetup::TypeTag type = eventsetup::TypeTag::findType( itType->second );
    if( type != defaultType ) {
      pProxyToToken pos=m_proxyToToken.find(buildName(iRecordKey.name(), type.name()));
      cond::Connection* c=conHandler.getConnection("inputdb");
      boost::shared_ptr<DataProxy> proxy(cond::ProxyFactory::get()->create( buildName(iRecordKey.name(), type.name() ), c, pos));
      if(0 != proxy.get()) {
	eventsetup::DataKey key( type, "");
	aProxyList.push_back(KeyedProxies::value_type(key,proxy));
      }
    }
  }
}

void 
PoolDBESSource::newInterval(const edm::eventsetup::EventSetupRecordKey& iRecordType,const edm::ValidityInterval& iInterval) 
{
  //std::cout<<"PoolDBESSource::newInterval"<<std::endl;
  LogDebug ("PoolDBESSource")<<"newInterval";
  invalidateProxies(iRecordType);
}

void 
PoolDBESSource::tagToToken( std::vector< std::pair<std::string, cond::TagMetadata> >& tagcollection ){
  std::vector< std::pair<std::string, cond::TagMetadata> >::iterator it;
  std::vector< std::pair<std::string, cond::TagMetadata> >::iterator itbeg=tagcollection.begin();
  std::vector< std::pair<std::string, cond::TagMetadata> >::iterator itend=tagcollection.end();
  for( it=itbeg;it!=itend;++it ){
    std::string tag=it->first;
    std::string recordName=it->second.recordname;
    std::string pfn=it->second.pfn;
    std::string iovToken;
    try{
      //std::cout<<"pfn "<<pfn<<std::endl;
      cond::Connection* connection=conHandler.getConnection(pfn);
      cond::CoralTransaction& coraldb=connection->coralTransaction(true);
      cond::MetaData metadata(coraldb);
      coraldb.start();
      iovToken=metadata.getToken(tag);
      if( iovToken.empty() ){
	throw cond::Exception("PoolDBESSource::tagToToken: tag "+tag+std::string(" has empty iov token") );
      }
      m_recordToIOV.insert(std::make_pair(recordName,iovToken));
      coraldb.commit();
    }catch(const cond::Exception&e ){
      throw e;
    }catch(const cms::Exception&e ){
      throw e;
    }
  }
}

