//
// Package:     CondCore/ESSources
// Module:      PoolDBESSource
//
// Description: <one line class summary>
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Zhen Xie
//
// system include files
#include "boost/shared_ptr.hpp"
#include "CondCore/ESSources/interface/PoolDBESSource.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/DBCommon/interface/ConfigSessionFromParameterSet.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "FWCore/Framework/interface/DataProxy.h"
#include "CondCore/PluginSystem/interface/ProxyFactory.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//#include <cstdlib>
#include "TagCollectionRetriever.h"
#include <exception>
//#include <iostream>
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

static void
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
   for (edmplugin::PluginManager::Infos::const_iterator itInfo = itFound->second.begin(),itInfoEnd = itFound->second.end(); itInfo != itInfoEnd; ++itInfo)
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

static cond::ConnectionHandler& conHandler=cond::ConnectionHandler::Instance();

PoolDBESSource::PoolDBESSource( const edm::ParameterSet& iConfig ) :
  m_session( new cond::DBSession )
{		
  //std::cout<<"PoolDBESSource::PoolDBESSource"<<std::endl;
  /*parameter set parsing and pool environment setting
   */
  std::string blobstreamerName("");
  if( iConfig.exists("BlobStreamerName") ){
    blobstreamerName=iConfig.getUntrackedParameter<std::string>("BlobStreamerName");
    blobstreamerName.insert(0,"COND/Services/");
    m_session->configuration().setBlobStreamer(blobstreamerName);
  }
  bool usetagDB=false;
  if( iConfig.exists("globaltag") ){
    usetagDB=true;
  }
 
  std::string userconnect;
  userconnect=iConfig.getParameter<std::string>("connect");
  //ignore "timetype" parameter
  //std::string timetype=iConfig.getParameter<std::string>("timetype");
  //std::cout<<"timetype "<<timetype<<std::endl;
  edm::ParameterSet connectionPset = iConfig.getParameter<edm::ParameterSet>("DBParameters"); 
  cond::ConfigSessionFromParameterSet configConnection(*m_session,connectionPset);
  m_session->open();
  
  if(!userconnect.empty()){
    conHandler.registerConnection(userconnect,*m_session,0);
  }

  fillRecordToTypeMap(m_recordToTypes);
  std::string lastRecordName;
  if( !usetagDB ){
    typedef std::vector< edm::ParameterSet > Parameters;
    Parameters toGet = iConfig.getParameter<Parameters>("toGet");
    std::string tagName;
    std::string recordName;
    std::string typeName;
    for(Parameters::iterator itToGet = toGet.begin(); itToGet != toGet.end(); ++itToGet ) {
      cond::TagMetadata m;
      if( itToGet->exists("connect") ){
	std::string userconnect=itToGet->getUntrackedParameter<std::string>("connect");
	m.pfn=userconnect;
	conHandler.registerConnection(m.pfn,*m_session,0);
      }else{
	m.pfn=userconnect;
      }
      if( itToGet->exists("label") ){
	m.labelname=itToGet->getUntrackedParameter<std::string>("label");
      }else{
	m.labelname="";
      }
      m.recordname = itToGet->getParameter<std::string>("record");
      m.tag=itToGet->getParameter<std::string>("tag");
      //tagName = itToGet->getParameter<std::string>("tag");
      std::multimap<std::string, std::string>::iterator itFound=m_recordToTypes.find(m.recordname);
      if(itFound == m_recordToTypes.end()){
	throw cond::Exception("NoRecord")<<" The record \""<<m.recordname<<"\" is not known by the PoolDBESSource";
      }
      m.objectname=itFound->second;
      std::string datumName=m.recordname+"@"+m.objectname+"@"+m.labelname;
      //std::cout<<"datumName "<<datumName<<std::endl;
      m_datumToToken.insert( std::make_pair<std::string,std::string>(datumName,"") );
      //fill in dummy tokens now, change in setIntervalFor
      DatumToToken::iterator pos=m_datumToToken.find(datumName);
      cond::Connection* c=conHandler.getConnection(m.pfn);
      conHandler.connect(m_session);
      boost::shared_ptr<edm::eventsetup::DataProxy> proxy(cond::ProxyFactory::get()->create(buildName(m.recordname, m.objectname), c, pos));
      edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType( m.recordname ) );
      if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag() ) {
	throw cond::Exception("NoRecord")<<"The record \""<< m.recordname <<"\" does not exist ";
      }
      if( lastRecordName != m.recordname ) {
	lastRecordName = m.recordname;
	findingRecordWithKey( recordKey );
	usingRecordWithKey( recordKey );
      }    
      if(! m_tagCollection.insert(m).second ){
	throw cond::Exception("PoolDBESSource::PoolDBESSource: "+m.tag+" from "+m.pfn+" already exist, cannot insert in the tag collection ");
      }
      //std::cout<<"tag collection size "<<m_tagCollection.size()<<std::endl;
    }
  }else{
    std::string globaltag=iConfig.getUntrackedParameter<std::string>("globaltag");
    cond::Connection* c=conHandler.getConnection(userconnect);
    conHandler.connect(m_session);
    cond::CoralTransaction& coraldb=c->coralTransaction();
    coraldb.start(true);
    this->fillTagCollectionFromDB(coraldb, globaltag);
    coraldb.commit();
    std::set<cond::TagMetadata>::iterator it;
    std::set<cond::TagMetadata>::iterator itBeg=m_tagCollection.begin();
    std::set<cond::TagMetadata>::iterator itEnd=m_tagCollection.end();
    for(it=itBeg; it!=itEnd; ++it){
      conHandler.registerConnection(it->pfn,*m_session,0);
    }
    conHandler.connect(m_session);
    for(it=itBeg;it!=itEnd;++it){
      //std::cout<<"recordname "<<it->second.recordname<<std::endl;
      //std::cout<<"objectname "<<it->second.objectname<<std::endl;
      //std::cout<<"labelname "<<it->second.labelname<<std::endl;
      std::multimap<std::string, std::string>::iterator itFound=m_recordToTypes.find(it->recordname);
      if(itFound == m_recordToTypes.end()){
      throw cond::Exception("NoRecord")<<" The record \""<<it->recordname<<"\" is not known by the PoolDBESSource";
      }
      std::string datumName = it->recordname+"@"+it->objectname+"@"+it->labelname;
      //std::cout<<"datumName "<<datumName<<std::endl;
      m_datumToToken.insert( make_pair(datumName,"") );
      //fill in dummy tokens now, change in setIntervalFor
      DatumToToken::iterator pos=m_datumToToken.find(datumName);
      cond::Connection* c=conHandler.getConnection(it->pfn); 
      boost::shared_ptr<edm::eventsetup::DataProxy> proxy(cond::ProxyFactory::get()->create(buildName(it->recordname, it->objectname), c, pos));
      edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType( it->recordname ) );
      if( recordKey.type() == edm::eventsetup::EventSetupRecordKey::TypeTag() ) {
	//record not found
	throw cond::Exception("NoRecord")<<"The record \""<< it->recordname <<"\" does not exist ";
      }
      if( lastRecordName != it->recordname ) {
	lastRecordName = it->recordname;
	findingRecordWithKey( recordKey );
	usingRecordWithKey( recordKey );
      }    
    }
    /*
      std::map< std::string, cond::TagMetadata >::iterator jt;
      std::map< std::string, cond::TagMetadata >::iterator jtBeg=m_tagCollection.begin();
      std::map< std::string, cond::TagMetadata >::iterator jtEnd=m_tagCollection.end();
      for( jt=jtBeg; jt!=jtEnd; ++jt ){
	std::cout<<"tag "<<jt->first<<std::endl;
	std::cout<<"pfn "<<jt->second.pfn<<std::endl;
	std::cout<<"recordname "<<jt->second.recordname<<std::endl;
	std::cout<<"objectname "<<jt->second.objectname<<std::endl;
	std::cout<<"labelname "<<jt->second.labelname<<std::endl;
      }
    */
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
PoolDBESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& iKey, const edm::IOVSyncValue& iTime, edm::ValidityInterval& oInterval ){
  //std::cout<<"PoolDBESSource::setIntervalFor"<<std::endl;
  //LogDebug ("PoolDBESSource")<<iKey.name();
  std::string recordname=iKey.name();
  std::string objectname("");
  std::string proxyname("");
  RecordToTypes::iterator itRec = m_recordToTypes.find( recordname );
  objectname=itRec->second;
  proxyname=buildName(recordname,objectname);
  if( itRec == m_recordToTypes.end() ) {
    LogDebug ("PoolDBESSource")<<"no valid record ";
    oInterval = edm::ValidityInterval::invalidInterval();
    return;
  }
  ///The first record of n proxies drives the IOV setting.
  ProxyToIOVInfo::iterator pos=m_proxyToIOVInfo.find(proxyname);
  if(pos==m_proxyToIOVInfo.end()){
    LogDebug ("PoolDBESSource")<<"no valid IOV found for proxy "<<proxyname;
    oInterval = edm::ValidityInterval::invalidInterval();
    return;
  }
  std::string leadingTag=pos->second.begin()->tag;
  std::string leadingToken=pos->second.begin()->token;
  std::string leadingLable=pos->second.begin()->label;
  cond::TimeType timetype=pos->second.begin()->timetype;
  //std::cout<<"leading tag "<<leadingTag<<std::endl;
  //std::cout<<"leading token "<<leadingToken<<std::endl;
  cond::Time_t abtime;
  if( timetype == cond::timestamp ){
    abtime=(cond::Time_t)iTime.time().value();
  }else{
    abtime=(cond::Time_t)iTime.eventID().run();
  }
  cond::Connection* c=conHandler.getConnection(pos->second.begin()->pfn);
  //std::cout<<"leading pfn "<< pos->second.front().pfn <<std::endl;
  cond::PoolTransaction& pooldb=c->poolTransaction();
  cond::IOVService iovservice(pooldb);  
  pooldb.start(true);
  std::ostringstream os;
  if( !iovservice.isValid(leadingToken,abtime) ){
    os<<abtime;
    //throw cond::noDataForRequiredTimeException("PoolDBESSource::setIntervalFor",iKey.name(),os.str());
    oInterval = edm::ValidityInterval::invalidInterval();
    return;
  }
  std::pair<cond::Time_t, cond::Time_t> validity=iovservice.validity(leadingToken,abtime);
  edm::IOVSyncValue start,stop;
  if( timetype == cond::timestamp ){
    start=edm::IOVSyncValue( edm::Timestamp(validity.first) );
    stop=edm::IOVSyncValue( edm::Timestamp(validity.second) );
  }else{
    start=edm::IOVSyncValue( edm::EventID(validity.first,0) );
    stop=edm::IOVSyncValue( edm::EventID(validity.second,0) );
  }
  oInterval = edm::ValidityInterval( start, stop );
  std::string payloadToken=iovservice.payloadToken(leadingToken,abtime);
  std::string datumName=recordname+"@"+objectname+"@"+leadingLable;
  m_datumToToken[datumName]=payloadToken;  
  pooldb.commit();

  std::set<cond::IOVInfo>::iterator itProxy;
  for(itProxy=pos->second.begin(); itProxy!=pos->second.end(); ++itProxy){
    if( (itProxy->label) != leadingLable){
      std::string datumName=recordname+"@"+objectname+"@"+itProxy->label;
      cond::Connection* c=conHandler.getConnection(itProxy->pfn);
      cond::PoolTransaction& labelpooldb=c->poolTransaction();
      cond::IOVService labeliovservice(labelpooldb);  
      labelpooldb.start(true);
      std::string payloadToken=labeliovservice.payloadToken(itProxy->token,abtime);
      labelpooldb.commit();
      m_datumToToken[datumName]=payloadToken;  
    }
  }
}

void 
PoolDBESSource::registerProxies(const edm::eventsetup::EventSetupRecordKey& iRecordKey , KeyedProxies& aProxyList) {
  //std::cout<<"registerProxies"<<std::endl;
  //LogDebug ("PoolDBESSource ")<<"registerProxies";
  //For each data type in this Record, create the proxy by dynamically loading it
  //loop over types in the same record
  std::string recordname=iRecordKey.name();
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
    //std::cout<<"proxyname "<<proxyname<<std::endl;
    ProxyToIOVInfo::iterator pProxyToIOVInfo=m_proxyToIOVInfo.find( proxyname );
    if ( pProxyToIOVInfo == m_proxyToIOVInfo.end() ) {
      //std::cout << "WARNING: Proxy not found " << proxyname<<std::endl;
      //LogDebug("PoolDBESSource ")<<"Proxy not found "<<proxyname<<", continue";
      //continue;
      throw cond::Exception(std::string("proxy ")+proxyname+"not found");
    }    
    for( std::set<cond::IOVInfo>::iterator it=pProxyToIOVInfo->second.begin();it!=pProxyToIOVInfo->second.end(); ++it ){
      //edm::eventsetup::IdTags iid( it->label.c_str() );
      std::string datumName=recordname+"@"+objectname+"@"+(it->label);
      std::map<std::string,std::string>::iterator pDatumToToken=m_datumToToken.find(datumName);
      if( pDatumToToken==m_datumToToken.end() ){
	throw cond::Exception(std::string("token not found for datum ")+datumName);
      }
      if( type != defaultType ) {
	cond::Connection* c=conHandler.getConnection(it->pfn);
	boost::shared_ptr<edm::eventsetup::DataProxy> proxy(cond::ProxyFactory::get()->create( proxyname ,c,pDatumToToken) );
	if(0 != proxy.get()) {
	  edm::eventsetup::DataKey key( type, edm::eventsetup::IdTags(it->label.c_str()) );
	  aProxyList.push_back(KeyedProxies::value_type(key,proxy));
	}
      }
    }
  }
}

void 
PoolDBESSource::newInterval(const edm::eventsetup::EventSetupRecordKey& iRecordType,const edm::ValidityInterval& iInterval) 
{
  //LogDebug ("PoolDBESSource")<<"newInterval";
  invalidateProxies(iRecordType);
}

void 
PoolDBESSource::fillRecordToIOVInfo(){
  std::set< cond::TagMetadata >::iterator it;
  std::set< cond::TagMetadata >::iterator itbeg=m_tagCollection.begin();
  std::set< cond::TagMetadata >::iterator itend=m_tagCollection.end();
  for( it=itbeg;it!=itend;++it ){
    std::string recordname=it->recordname;
    std::string objectname=it->objectname;
    std::string proxyname=buildName(recordname,objectname);
    cond::IOVInfo iovInfo;
    iovInfo.tag=it->tag;
    iovInfo.pfn=it->pfn;
    iovInfo.label=it->labelname;
    
    cond::Connection* connection=conHandler.getConnection(iovInfo.pfn);
    cond::CoralTransaction& coraldb=connection->coralTransaction();
    cond::MetaData metadata(coraldb);
    coraldb.start(true);
    cond::MetaDataEntry result;
    metadata.getEntryByTag(iovInfo.tag,result);
    iovInfo.token=result.iovtoken;
    iovInfo.timetype=result.timetype;
    coraldb.commit();
    if( iovInfo.token.empty() ){
      throw cond::Exception("PoolDBESSource::fillrecordToIOVInfo: tag "+iovInfo.tag+std::string(" has empty iov token meaning requested tag or data do not exist") );
    }
    std::map<std::string,std::set<cond::IOVInfo> >::iterator pos=m_proxyToIOVInfo.find(proxyname);
    if( pos!= m_proxyToIOVInfo.end() ){
      //unclean tag rename protection. Ignore if same iovInfo found again  
      if( ! pos->second.insert(iovInfo).second ){
	std::set<cond::IOVInfo>::iterator t  = pos->second.find(iovInfo);
	edm::LogWarning("PoolDBESSource")<<"Warning: ignore registering data proxy for tag "<<iovInfo.tag<<" since it contains identical information as tag "<<t->tag <<" : "<<"iovtoken "<<t->token<<" provenance "<<t->pfn<<" label "<<t->label;
      }
    }else{
      std::set<cond::IOVInfo> infos;
      infos.insert(iovInfo);
      m_proxyToIOVInfo.insert(std::make_pair<std::string,std::set<cond::IOVInfo> >(proxyname,infos));
    }
  }
}
void 
PoolDBESSource::fillTagCollectionFromDB( cond::CoralTransaction& coraldb, 
					 const std::string& roottag ){
  cond::TagCollectionRetriever tagRetriever( coraldb );
  tagRetriever.getTagCollection(roottag,m_tagCollection);
}
