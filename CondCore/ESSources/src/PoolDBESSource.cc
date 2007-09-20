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
#include <sstream>
#include <cstdlib>
#include "TagCollectionRetriever.h"
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
  m_session( new cond::DBSession )
{		
  //std::cout<<"PoolDBESSource::PoolDBESSource"<<std::endl;
  /*parameter set parsing and pool environment setting
   */
  bool usetagDB=false;
  if( iConfig.exists("globaltag") ){
    std::cout<<"exists global tag parameter"<<std::endl;
    usetagDB=true;
  }
  std::string connect=iConfig.getParameter<std::string>("connect"); 
  std::string timetype=iConfig.getUntrackedParameter<std::string>("timetype","runnumber");
  std::cout<<"connect "<<connect<<std::endl;
  std::cout<<"timetype "<<timetype<<std::endl;
  edm::ParameterSet connectionPset = iConfig.getParameter<edm::ParameterSet>("DBParameters"); 
  cond::ConfigSessionFromParameterSet configConnection(*m_session,connectionPset);
  m_session->open();
  if( connect.find("sqlite_fip:") != std::string::npos ){
    cond::FipProtocolParser p;
    connect=p.getRealConnect(connect);
  }
  std::cout<<"using connect "<<connect<<std::endl;
  ///handle frontier cache refresh
  if( connect.rfind("frontier://") != std::string::npos){
    connect=this->setupFrontier(connect);
  }
  conHandler.registerConnection(connect,connect,0);
  fillRecordToTypeMap(m_recordToTypes);
  std::cout<<"filled record to type map"<<std::endl;
  if( !usetagDB ){
    std::cout<<"not using tag db"<<std::endl;
    typedef std::vector< edm::ParameterSet > Parameters;
    Parameters toGet = iConfig.getParameter<Parameters>("toGet");
    std::string tagName;
    std::string recordName;
    std::string typeName;
    std::string lastRecordName;
    for(Parameters::iterator itToGet = toGet.begin(); itToGet != toGet.end(); ++itToGet ) {
      cond::TagMetadata m;
      if( itToGet->exists("connect") ){
	std::string connect=itToGet->getUntrackedParameter<std::string>("connect");
	if( connect.find("sqlite_fip:") != std::string::npos ){
	  cond::FipProtocolParser p;
	  connect=p.getRealConnect(connect);
	}
	m.pfn=connect;
	conHandler.registerConnection(m.pfn,m.pfn,0);
	//std::cout<<"using local pfn "<<m.pfn<<std::endl;
      }else{
	m.pfn=connect;
	//std::cout<<"using global pfn "<<m.pfn<<std::endl;
      }
      if( itToGet->exists("timetype") ){
	m.timetype=itToGet->getUntrackedParameter<std::string>("timetype");
	//std::cout<<"using local timetype "<<m.timetype<<std::endl;
      }else{
	m.timetype=timetype;
	//std::cout<<"using global timetype "<<m.timetype<<std::endl;
      }
      if( itToGet->exists("label") ){
	m.labelname=itToGet->getUntrackedParameter<std::string>("label");
      }else{
	m.labelname="";
      }
      m.recordname = itToGet->getParameter<std::string>("record");
      tagName = itToGet->getParameter<std::string>("tag");
      //std::cout<<"requested record "<<m.recordname<<std::endl;
      //std::cout<<"requested tag "<<tagName<<std::endl;
      
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
      ProxyToToken::iterator pos=m_proxyToToken.find(proxyName);
      cond::Connection* c=conHandler.getConnection(m.pfn);
      conHandler.connect(m_session);
      boost::shared_ptr<edm::eventsetup::DataProxy> proxy(cond::ProxyFactory::get()->create(buildName(m.recordname, m.objectname), c, pos));
      edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType( m.recordname ) );
      if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag() ) {
	//record not found
	throw cond::Exception("NoRecord")<<"The record \""<< m.recordname <<"\" does not exist ";
      }
      //recordToTag.push_back(std::make_pair(m.recordname, tagName));
      if( lastRecordName != m.recordname ) {
	lastRecordName = m.recordname;
	findingRecordWithKey( recordKey );
	usingRecordWithKey( recordKey );
      }    
      m_tagCollection.insert(std::make_pair<std::string,cond::TagMetadata>(tagName,m));
    }
  }else{
    std::cout<<"here"<<std::endl;
    std::string globaltag=iConfig.getParameter<std::string>("globaltag");
    std::cout<<"globaltag "<<globaltag<<std::endl;
    cond::Connection* c=conHandler.getConnection(connect);
    conHandler.connect(m_session);
    cond::CoralTransaction& coraldb=c->coralTransaction(true);
    coraldb.start();
    this->fillTagCollectionFromDB(coraldb, globaltag);
    coraldb.commit();
  }
  this->fillRecordToIOVInfo();
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
  RecordToIOVInfo::iterator itIOV = m_recordToIOVInfo.find( iKey.name() );
  if( itIOV == m_recordToIOVInfo.end() ){
    LogDebug ("PoolDBESSource")<<"no valid IOV found for record "<<iKey.name();
    oInterval = edm::ValidityInterval::invalidInterval();
    return;
  }
  std::string iovToken=itIOV->second.token;
  std::string iovtag=itIOV->second.tag;
  std::string payloadToken;
  cond::Time_t abtime;
  std::ostringstream os;

  TagCollection::iterator tagit=m_tagCollection.find(iovtag);
  if(tagit == m_tagCollection.end() ){
    throw cond::Exception("so strange!!");
  }
  if( tagit->second.timetype == "timestamp" ){
    abtime=(cond::Time_t)iTime.time().value();
  }else{
    abtime=(cond::Time_t)iTime.eventID().run();
  }
  //valid time check
  //check if current run exceeds iov upperbound
  /*standalone    
   */
  cond::Connection* c=conHandler.getConnection(tagit->second.pfn);
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
  if( tagit->second.timetype == "timestamp" ){
    start=edm::IOVSyncValue( edm::Timestamp(validity.first) );
  }else{
    start=edm::IOVSyncValue( edm::EventID(validity.first,0) );
  }
  edm::IOVSyncValue stop;
  if( tagit->second.timetype == "timestamp" ){
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
  /*
  std::cout<<"setting itRec->first "<<itRec->first<<std::endl;
  std::cout<<"setting itRec->second "<<itRec->second<<std::endl;
  std::cout<<"payloadToken "<< payloadToken<<std::endl;
  std::cout<<"buildProxy "<<buildName(itRec->first ,itRec->second)<<std::endl;
  */
  m_proxyToToken[buildName(itRec->first ,itRec->second)]=payloadToken;  
}   

void 
PoolDBESSource::registerProxies(const edm::eventsetup::EventSetupRecordKey& iRecordKey , KeyedProxies& aProxyList) 
{
  //LogDebug ("PoolDBESSource ")<<"registerProxies";
  //using namespace edm;
  //using namespace edm::eventsetup;
  //using namespace std;
  //std::cout <<"registering Proxies for "<< iRecordKey.name() << std::endl;
  //For each data type in this Record, create the proxy by dynamically loading it
  std::string recordname=iRecordKey.name();
  RecordToIOVInfo::iterator itRec=m_recordToIOVInfo.find(recordname);
  std::string tagname=itRec->second.tag;
  TagCollection::iterator itTag=m_tagCollection.find(tagname);
  std::string labelname=itTag->second.labelname;
  cond::Connection* c=conHandler.getConnection(itTag->second.pfn);
  std::pair< RecordToTypes::iterator,RecordToTypes::iterator > typeItrs = m_recordToTypes.equal_range( recordname );
  //loop over types in the same record
  for( RecordToTypes::iterator itType = typeItrs.first; itType != typeItrs.second; ++itType ) {
    //std::cout<<"Entering loop PoolDBESSource::registerProxies"<<std::endl;
    //std::cout<<std::string("   ") + itType->second <<std::endl;
    static edm::eventsetup::TypeTag defaultType;
    edm::eventsetup::TypeTag type = edm::eventsetup::TypeTag::findType( itType->second );
    edm::eventsetup::IdTags iid(labelname.c_str());
    if( type != defaultType ) {
      ProxyToToken::iterator pos=m_proxyToToken.find(buildName(recordname, type.name()));
      boost::shared_ptr<edm::eventsetup::DataProxy> proxy(cond::ProxyFactory::get()->create( buildName(recordname, type.name() ), c, pos));
      if(0 != proxy.get()) {
	edm::eventsetup::DataKey key( type, iid );
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
PoolDBESSource::fillRecordToIOVInfo(){
  std::map< std::string, cond::TagMetadata >::iterator it;
  std::map< std::string, cond::TagMetadata >::iterator itbeg=m_tagCollection.begin();
  std::map< std::string, cond::TagMetadata >::iterator itend=m_tagCollection.end();
  for( it=itbeg;it!=itend;++it ){
    std::string pfn=it->second.pfn;
    std::string recordname=it->second.recordname;
    cond::IOVInfo iovInfo;
    iovInfo.tag=it->first;
    try{
      //std::cout<<"pfn "<<pfn<<std::endl;
      cond::Connection* connection=conHandler.getConnection(pfn);
      cond::CoralTransaction& coraldb=connection->coralTransaction(true);
      cond::MetaData metadata(coraldb);
      coraldb.start();
      iovInfo.token=metadata.getToken(iovInfo.tag);
      if( iovInfo.token.empty() ){
	throw cond::Exception("PoolDBESSource::tagToToken: tag "+iovInfo.tag+std::string(" has empty iov token") );
      }
      m_recordToIOVInfo.insert( std::make_pair<std::string,cond::IOVInfo>(recordname,iovInfo) );
      coraldb.commit();
    }catch(const cond::Exception&e ){
      throw e;
    }catch(const std::exception&e ){
      throw e;
    }
  }
}
std::string
PoolDBESSource::setupFrontier(const std::string& frontierconnect){ 
  //Mark tables that need to not be cached (always refreshed)
  //strip off the leading protocol:// and trailing schema name from connect
  edm::Service<edm::SiteLocalConfig> localconfservice;
  if( !localconfservice.isAvailable() ){
    throw cms::Exception("edm::SiteLocalConfigService is not available");       
  }
  std::string realconnect=localconfservice->lookupCalibConnect(frontierconnect);
  std::string::size_type startRefresh = realconnect.find("://");
  if (startRefresh != std::string::npos){
    startRefresh += 3;
  }
  std::string::size_type endRefresh = realconnect.rfind("/", std::string::npos);
  std::string refreshConnect;
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
  //get handle to webCacheControl()
  m_session->webCacheControl().refreshTable( refreshConnect,cond::IOVNames::iovTableName() );
  m_session->webCacheControl().refreshTable( refreshConnect,cond::IOVNames::iovDataTableName() );
  m_session->webCacheControl().refreshTable( refreshConnect,cond::MetaDataNames::metadataTable() );
  return realconnect;
}
void 
PoolDBESSource::fillTagCollectionFromDB( cond::CoralTransaction& coraldb, 
					 const std::string& roottag ){
  cond::TagCollectionRetriever tagRetriever( coraldb );
  tagRetriever.getTagCollection(roottag,m_tagCollection);
  //m_tagCollection.insert(std::make_pair<std::string,cond::TagMetadata>(tagName,m));
}
