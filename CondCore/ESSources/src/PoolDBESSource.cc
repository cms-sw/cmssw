// -*- C++ -*-
//
// Package:     CondCore/ESSources
// Module:      PoolDBESSource
//
// Author:      Zhen Xie
//

// system include files
#include "boost/shared_ptr.hpp"
// user include files
#include "CondCore/ESSources/interface/PoolDBESSource.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/ConnectMode.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/DBCommon/interface/RelationalStorageManager.h"
#include "CondCore/DBCommon/interface/PoolStorageManager.h"
#include "CondCore/DBCommon/interface/ConfigSessionFromParameterSet.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/src/ServiceLoader.h"
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
#include "CondCore/DBCommon/interface/FipProtocolParser.h"
#include <exception>
//#include <iostream>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FileCatalog/IFileCatalog.h"
//#include <sstream>
//#include <cstdlib>
//namespace fs = boost::filesystem;
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
PoolDBESSource::PoolDBESSource( const edm::ParameterSet& iConfig ) :
  m_session( 0 ), 
  m_timetype(iConfig.getParameter<std::string>("timetype") ),
  m_connected( false )
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
  //std::cout<<"using connect "<<connect<<std::endl;
  m_session=new cond::DBSession(true);
  std::string blobstreamerName("");
  if( iConfig.exists("BlobStreamerName") ){
    blobstreamerName=iConfig.getUntrackedParameter<std::string>("BlobStreamerName");
    blobstreamerName.insert(0,"COND/Services/");
    //std::cout<<"blobstreamerName "<<blobstreamerName<<std::endl;
    m_session->sessionConfiguration().setBlobStreamer(blobstreamerName);
  }
  edm::ParameterSet connectionPset = iConfig.getParameter<edm::ParameterSet>("DBParameters"); 
  using namespace edm;
  using namespace edm::eventsetup;  
  fillRecordToTypeMap(m_recordToTypes);
  //parsing record to tag
  std::vector< std::pair<std::string,std::string> > recordToTag;
  typedef std::vector< ParameterSet > Parameters;
  Parameters toGet = iConfig.getParameter<Parameters>("toGet");
  if(0==toGet.size()){
    throw cond::Exception("Configuration") <<" The \"toGet\" parameter is empty, please specify what (Record, tag) pairs you wish to retrieve\n"
					   <<" or use the record name \"all\" to have your tag be used to retrieve all known Records\n";
  }
  std::string lastRecordName;
  for(Parameters::iterator itToGet = toGet.begin(); itToGet != toGet.end(); ++itToGet ) {
    std::string recordName = itToGet->getParameter<std::string>("record");
    std::string tagName = itToGet->getParameter<std::string>("tag");
    std::string labelname("");
    if( itToGet->exists("label") ){
      labelname=itToGet->getUntrackedParameter<std::string>("label");
    }
    //load proxy code now to force in the Record code
    std::multimap<std::string, std::string>::iterator itFound=m_recordToTypes.find(recordName);
    if(itFound == m_recordToTypes.end()){
      throw cond::Exception("NoRecord")<<" The record \""<<recordName<<"\" is not known by the PoolDBESSource";
    }
    std::string typeName = itFound->second;
    std::string proxyName = buildName(recordName,typeName);
    std::string datumName=recordName+"@"+typeName+"@"+labelname;
    m_datumToToken.insert( std::make_pair<std::string,std::string>(datumName,"") );
    cond::TagMetadata m;
    m.labelname=labelname;
    m.recordname = recordName;
    m.objectname = typeName;
    m_tagCollection.insert(std::make_pair<std::string,cond::TagMetadata>(tagName,m));
    //fill in dummy tokens now, change in setIntervalFor
    DatumToToken::iterator pos=m_datumToToken.find(datumName);
    boost::shared_ptr<DataProxy> proxy(cond::ProxyFactory::get()->create(proxyName,m_pooldb,pos));
    eventsetup::EventSetupRecordKey recordKey(eventsetup::EventSetupRecordKey::TypeTag::findType( recordName ) );
    if( recordKey.type() == eventsetup::EventSetupRecordKey::TypeTag() ) {
      //record not found
      throw cond::Exception("NoRecord")<<"The record \""<< recordName <<"\" does not exist ";
    }
    //recordToTag.push_back(std::make_pair(recordName, tagName));
    if( lastRecordName != recordName ) {
      lastRecordName = recordName;
      findingRecordWithKey( recordKey );
      usingRecordWithKey( recordKey );
    }
  }
  cond::ConfigSessionFromParameterSet configConnection(*m_session,connectionPset);
  m_session->open();
  ///handle frontier cache refresh
  std::string proto("frontier://");
  std::string::size_type fpos=connect.find(proto);
  if( fpos!= std::string::npos){
    unsigned int nslash=this->countslash(connect.substr(proto.size(),connect.size()-fpos));
    //if( nslash!=1 && nslash!=2) {
    //  throw cms::Exception("connect string "+connect+" has bad format");
    //}
    //Mark tables that need to not be cached (always refreshed)
    //strip off the leading protocol:// and trailing schema name from connect
    if(nslash==1){
      //frontier connect string need site local translation
      edm::Service<edm::SiteLocalConfig> localconfservice;
      if( !localconfservice.isAvailable() ){
	throw cms::Exception("edm::SiteLocalConfigService is not available"); 
      }
      connect=localconfservice->lookupCalibConnect(connect);
    }
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
    seal::IHandle<coral::IConnectionService> connSvc=m_session->serviceLoader().context()->query<coral::IConnectionService>( "CORAL/Services/ConnectionService" );
    //get handle to webCacheControl()
    connSvc->webCacheControl().refreshTable( refreshConnect,cond::IOVNames::iovTableName() );
    connSvc->webCacheControl().refreshTable( refreshConnect,cond::IOVNames::iovDataTableName() );
    connSvc->webCacheControl().refreshTable( refreshConnect,cond::MetaDataNames::metadataTable() );
       }
  m_con=connect;
  std::string catconnect="pfncatalog_memory://POOL_RDBMS?";
  catconnect.append(m_con);
  m_pooldb=new cond::PoolStorageManager(m_con,catconnect,m_session);
  if(m_timetype=="timestamp"){
    m_iovservice=new cond::IOVService(*m_pooldb,cond::timestamp);
  }else{
    m_iovservice=new cond::IOVService(*m_pooldb,cond::runnumber);
  }
  this->fillRecordToIOVInfo();
  //this->tagToToken(recordToTag);
}
PoolDBESSource::~PoolDBESSource()
{
  // std::cout<<"PoolDBESSource::~PoolDBESSource"<<std::endl;
  if( m_session->isActive() ){
    m_pooldb->disconnect();
    m_session->close();
  }
  delete m_iovservice;
  delete m_pooldb;
  delete m_session;
}
//
// member functions
//
void 
PoolDBESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& iKey, const edm::IOVSyncValue& iTime, edm::ValidityInterval& oInterval ) {
  //LogDebug ("PoolDBESSource")<<iKey.name();
  //std::cout<<"PoolDBESSource::setIntervalFor "<< iKey.name() <<" at time "<<iTime.eventID().run()<<std::endl;
  std::string recordname=iKey.name();
  std::string objectname("");
  std::string proxyname("");
  std::string payloadToken("");
  RecordToTypes::iterator itRec = m_recordToTypes.find( recordname );
  objectname=itRec->second;
  proxyname=buildName(recordname,objectname);
  if( itRec == m_recordToTypes.end() ) {
    //no valid record
    LogDebug ("PoolDBESSource")<<"no valid record ";
    //std::cout<<"no valid record "<<std::endl;
    oInterval = edm::ValidityInterval::invalidInterval();
    return;
  }
  //std::cout<<"recordToIOV size "<<m_recordToIOV.size()<<std::endl;
  ProxyToIOVInfo::iterator pos = m_proxyToIOVInfo.find( proxyname );
  if(pos==m_proxyToIOVInfo.end()){
    LogDebug ("PoolDBESSource")<<"no valid IOV found for proxy "<<proxyname;
    oInterval = edm::ValidityInterval::invalidInterval();
    return;
  }
  std::string leadingTag=pos->second.front().tag;
  std::string leadingToken=pos->second.front().token;
  std::string leadingLable=pos->second.front().label;
  cond::Time_t abtime;
  std::ostringstream os;
  if( m_timetype == "timestamp" ){
    abtime=(cond::Time_t)iTime.time().value();
  }else{
    abtime=(cond::Time_t)iTime.eventID().run();
  }
  //valid time check
  //check if current run exceeds iov upperbound
  m_pooldb->startTransaction(true);
  if( !m_iovservice->isValid(leadingToken,abtime) ){
    os<<abtime;
    throw cond::noDataForRequiredTimeException("PoolDBESSource::setIntervalFor",iKey.name(),os.str());
  }
  std::pair<cond::Time_t, cond::Time_t> validity=m_iovservice->validity(leadingToken,abtime);
  
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
  payloadToken=m_iovservice->payloadToken(leadingToken,abtime);
  std::string datumName=recordname+"@"+objectname+"@"+leadingLable;
  m_datumToToken[datumName]=payloadToken;  

  std::vector<cond::IOVInfo>::iterator itProxy;
  for(itProxy=pos->second.begin(); itProxy!=pos->second.end(); ++itProxy){
    if( (itProxy->label) != leadingLable){
      std::string datumName=recordname+"@"+objectname+"@"+itProxy->label;
      payloadToken=m_iovservice->payloadToken(itProxy->token,abtime);
      m_datumToToken[datumName]=payloadToken;  
    }
  }
  m_pooldb->commit();
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
  //std::cout<<"recordname "<<recordname<<std::endl;
  std::string objectname("");
  std::string proxyname("");
  std::pair< RecordToTypes::iterator,RecordToTypes::iterator > typeItrs = m_recordToTypes.equal_range( recordname );
  for( RecordToTypes::iterator itType = typeItrs.first; itType != typeItrs.second; ++itType ) {
    static const edm::eventsetup::TypeTag defaultType;
    edm::eventsetup::TypeTag type = edm::eventsetup::TypeTag::findType(itType->second);
    if(defaultType == type ) {
      //std::cout <<"WARNING: unknown type '"<<itType->second<<"'"<<std::endl;
      LogDebug("PoolDBESSource ")<<"unknown type '" <<itType->second<<"', continue";
      continue;
    }
    objectname=type.name();
    proxyname=buildName(recordname,objectname);
    ProxyToIOVInfo::iterator pProxyToIOVInfo=m_proxyToIOVInfo.find( proxyname );
    if ( pProxyToIOVInfo == m_proxyToIOVInfo.end() ) {
      //std::cout << "WARNING: Proxy not found " << proxyname<<std::endl;
      LogDebug("PoolDBESSource ")<<"Proxy not found "<<proxyname<<", continue";
      continue;
    }
    for( std::vector<cond::IOVInfo>::iterator it=pProxyToIOVInfo->second.begin();it!=pProxyToIOVInfo->second.end(); ++it ){
      //edm::eventsetup::IdTags iid( it->label.c_str() );
      std::string datumName=recordname+"@"+objectname+"@"+(it->label);
      std::map<std::string,std::string>::iterator pDatumToToken=m_datumToToken.find(datumName);
      if( type != defaultType ) {
	boost::shared_ptr<edm::eventsetup::DataProxy> proxy(cond::ProxyFactory::get()->create( proxyname ,m_pooldb,pDatumToToken) );
	if(0 != proxy.get()) {
	  edm::eventsetup::DataKey key( type,edm::eventsetup::IdTags(it->label.c_str()));
	  aProxyList.push_back(KeyedProxies::value_type(key,proxy));
	}
      }
    }
  }
  if( !m_connected ){
    m_pooldb->connect();
    m_connected=true;
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
  try{
    for( it=itbeg;it!=itend;++it ){
      std::string recordname=it->second.recordname;
      std::string objectname=it->second.objectname;
      std::string proxyname=buildName(recordname,objectname);
      cond::IOVInfo iovInfo;
      iovInfo.tag=it->first;
      iovInfo.label=it->second.labelname;
      cond::RelationalStorageManager coraldb(m_con,m_session);
      cond::MetaData metadata(coraldb);
      coraldb.connect(cond::ReadOnly);
      coraldb.startTransaction(true);
      iovInfo.token=metadata.getToken(iovInfo.tag);
      coraldb.commit();
      coraldb.disconnect();
      if( iovInfo.token.empty() ){
	throw cond::Exception("PoolDBESSource::fillrecordToIOVInfo: tag "+iovInfo.tag+std::string(" has empty iov token") );
      }
 
      std::map<std::string,std::vector<cond::IOVInfo> >::iterator pos=m_proxyToIOVInfo.find(proxyname);
      if( pos!= m_proxyToIOVInfo.end() ){
	pos->second.push_back(iovInfo);
      }else{
       	std::vector<cond::IOVInfo> infos;
	infos.push_back(iovInfo);
	m_proxyToIOVInfo.insert(std::make_pair<std::string,std::vector<cond::IOVInfo> >(proxyname,infos));
      }
    }
  }catch(const cond::Exception&e ){
    throw e;
  }catch(const std::exception&e ){
    throw e;
  }
}
unsigned int
PoolDBESSource::countslash(const std::string& input)const{
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


