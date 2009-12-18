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
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondFormats/Common/interface/Time.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/ConvertIOVSyncValue.h"

// #include "FWCore/Framework/interface/DataProxy.h"
#include "CondCore/PluginSystem/interface/DataProxy.h"
#include "CondCore/PluginSystem/interface/ProxyFactory.h"
#include "CondCore/IOVService/interface/PayloadProxy.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//#include <cstdlib>
#include "CondCore/TagCollection/interface/TagCollectionRetriever.h"
#include <exception>
//#include <iostream>

#include "CondFormats/Common/interface/TimeConversions.h"


namespace {
  /* utility ot build the name of the plugin corresponding to a given record
     se PluginSystem
   */
  std::string
  buildName( const std::string& iRecordName) {
    return iRecordName+"@NewProxy";
  }


  /* utility class to return a IOVs associated to a given "name"
     This implementation return the IOV associated to a record...
     It is essentialy a workaround to get the full IOV out of the tag colector
     that is not accessible as hidden in the ESSource
     FIXME: need to support label??
   */
  class CondGetterFromESSource : public cond::CondGetter {
  public:
    CondGetterFromESSource(PoolDBESSource::ProxyMap const & ip) : m_proxies(ip){}
    virtual ~CondGetterFromESSource(){}

    cond::IOVProxy get(std::string name) const {
      PoolDBESSource::ProxyMap::const_iterator p = m_proxies.find(name);
      if ( p != m_proxies.end())
	return (*p).second->proxy()->iov();
      return cond::IOVProxy();
    }

    PoolDBESSource::ProxyMap const & m_proxies;
  };

  // dump the state of a DataProxy
  void dumpInfo(std::ostream & out, std::string const & recName, cond::DataProxyWrapperBase const & proxy) {
    cond::SequenceState state(proxy.proxy()->iov().state());
    out << recName << " / " << proxy.label() << ": " 
	<< proxy.connString() << ", " << proxy.tag()   << "\n  "
	<< state.size() << ", " << state.revision()  << ", "
	<< cond::time::to_boost(state.timestamp())     << "\n  "
	<< state.comment();

  }


}


/*
 *  config Param
 *  RefreshEachRun: if true will refresh the IOV at each new run (or lumiSection)
 *  DumpStat: if true dump the statistics of all DataProxy (currently on cout)
 *  DBParameters: configuration set of the connection
 *  globaltag: The GlobalTag
 *  toGet: list of record label tag connection-string to add/overwrite the content of the global-tag
 */
PoolDBESSource::PoolDBESSource( const edm::ParameterSet& iConfig ) :
  m_connection(), 
  lastRun(0),  // for the refresh
  doRefresh(iConfig.getUntrackedParameter<bool>("RefreshEachRun",false)),
  doDump(iConfig.getUntrackedParameter<bool>("DumpStat",false))
{
   Stats s = {0,0,0,0,0};
   stats=s;	
  //std::cout<<"PoolDBESSource::PoolDBESSource"<<std::endl;
  /*parameter set parsing and pool environment setting
   */


  std::string userconnect;
  userconnect=iConfig.getParameter<std::string>("connect");
  edm::ParameterSet connectionPset = iConfig.getParameter<edm::ParameterSet>("DBParameters");
  m_connection.configuration().setParameters( connectionPset );
  m_connection.configure();
  
  cond::DbSession session = m_connection.createSession();
  std::string blobstreamerName("");
  if( iConfig.exists("BlobStreamerName") ){
    blobstreamerName=iConfig.getUntrackedParameter<std::string>("BlobStreamerName");
    blobstreamerName.insert(0,"COND/Services/");
    session.setBlobStreamingService(blobstreamerName);
  }
  if(!userconnect.empty())
    session.open( userconnect );
  
  std::string globaltag;
  if( iConfig.exists("globaltag")) globaltag=iConfig.getParameter<std::string>("globaltag");

  // load additional record/tag info it will overwrite the global tag
  std::map<std::string,cond::TagMetadata> replacement;
  if( iConfig.exists("toGet") ){
    typedef std::vector< edm::ParameterSet > Parameters;
    Parameters toGet = iConfig.getParameter<Parameters>("toGet");
    for(Parameters::iterator itToGet = toGet.begin(); itToGet != toGet.end(); ++itToGet ) {
      cond::TagMetadata nm;
      nm.recordname=itToGet->getParameter<std::string>("record");
      nm.labelname=itToGet->getUntrackedParameter<std::string>("label","");
      nm.tag=itToGet->getParameter<std::string>("tag");
      nm.pfn=itToGet->getUntrackedParameter<std::string>("connect",userconnect);
      //	nm.objectname=itFound->second;
      std::string k=nm.recordname+"@"+nm.labelname;
      replacement.insert(std::make_pair<std::string,cond::TagMetadata>(k,nm));
    }
  }

  // get the global tag, merge with "replacement" store in "tagCollection"
  session.transaction().start(true);
  fillTagCollectionFromDB(session, globaltag,replacement);
  session.transaction().commit();


  TagCollection::iterator it;
  TagCollection::iterator itBeg=m_tagCollection.begin();
  TagCollection::iterator itEnd=m_tagCollection.end();
  //for(it=itBeg; it!=itEnd; ++it){
  //  cond::ConnectionHandler::Instance().registerConnection(it->pfn,m_session,0);
  //}

  //  cond::ConnectionHandler::Instance().connect(&m_session);
  for(it=itBeg;it!=itEnd;++it){

    //open db get tag info (i.e. the IOV token...)
    cond::DbSession nsess = m_connection.createSession();
    nsess.open( it->pfn);
    cond::MetaData metadata(nsess);
    nsess.transaction().start(true);
    cond::MetaDataEntry result;
    metadata.getEntryByTag(it->tag,result);
    nsess.transaction().commit();

    /* load DataProxy Plugin (it is strongly typed due to EventSetup ideosyncrasis)
     * construct proxy
     * contrary to EventSetup the "object-name" is not used as identifier: multipl entries in a record are
     * dinstinguished only by their label...
     */
    cond::DataProxyWrapperBase * pb =  
      cond::ProxyFactory::get()->create(buildName(it->recordname), nsess,
					cond::DataProxyWrapperBase::Args(result.iovtoken, it->labelname));
    // owenship...
    ProxyP proxy(pb);
    // add more info (not in constructor to overcome the mess/limitation of the templated factory
    proxy->addInfo(it->pfn, it->tag);
    //  instert in the map
    m_proxies.insert(std::make_pair(it->recordname, proxy));
  }

  // expose all other tags to the Proxy! 
  CondGetterFromESSource visitor(m_proxies);
  ProxyMap::iterator b= m_proxies.begin();
  ProxyMap::iterator e= m_proxies.end();
  for (;b!=e;b++) {
    (*b).second->proxy()->loadMore(visitor);

    /// required by eventsetup
    edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType( (*b).first ) );
    if( recordKey.type() != edm::eventsetup::EventSetupRecordKey::TypeTag() ) {
      findingRecordWithKey( recordKey );
      usingRecordWithKey( recordKey );
    }
  }

  stats.nData=m_proxies.size();
}


PoolDBESSource::~PoolDBESSource() {
  //dump info FIXME: find a more suitable place...
  if (doDump) {
    std::cout << "PoolDBESSource Statistics" << std::endl
	      << "DataProxy " << stats.nData
	      <<" setInterval " << stats.nSet
	      <<" Runs " << stats.nRun
	      <<" Refresh " << stats.nRefresh
	      <<" Actual Refresh " << stats.nActualRefresh;
    std::cout << std::endl;
    std::cout << "Proxy Statistics" << std::endl
	      << "proxy " << cond::BasePayloadProxy::stats.nProxy
	      << " make " << cond::BasePayloadProxy::stats.nMake
	      << " load " << cond::BasePayloadProxy::stats.nLoad;
    std::cout << std::endl;


    ProxyMap::iterator b= m_proxies.begin();
    ProxyMap::iterator e= m_proxies.end();
    for (;b!=e;b++) {
      dumpInfo(std::cout,(*b).first,*(*b).second);
      std::cout << "\n" << std::endl;
    }
  }
}


//
// invoked by EventSetUp: for a given record return the smallest IOV for which iTime is valid
// limit to next run/lumisection of Refresh is required
//
void 
PoolDBESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& iKey, const edm::IOVSyncValue& iTime, edm::ValidityInterval& oInterval ){

  stats.nSet++;

  std::string recordname=iKey.name();
  oInterval = edm::ValidityInterval::invalidInterval();
  
  //FIXME use equal_range
  ProxyMap::const_iterator b = m_proxies.lower_bound(recordname);
  ProxyMap::const_iterator e = m_proxies.upper_bound(recordname);
  if ( b == e) {
    LogDebug ("PoolDBESSource")<<"no DataProxy (Pluging) found for record "<<recordname;
    return;
  }

  // compute the smallest interval (assume all objects have the same timetype....)
  cond::ValidityInterval recordValidity(0,cond::TIMELIMIT);
  cond::TimeType timetype;
  bool userTime=true;
  for (ProxyMap::const_iterator p=b;p!=e;++p) {
    // refresh if required...
    if (doRefresh)  { 
      stats.nActualRefresh += (*p).second->proxy()->refresh(); 
      stats.nRefresh++;
    }
    
    {
      // not required anymore, keep here for the time being
      if(iTime.eventID().run()!=lastRun) {
	lastRun=iTime.eventID().run();
	stats.nRun++;
      }
    }
    
    
    timetype = (*p).second->proxy()->timetype();
    
    cond::Time_t abtime = cond::fromIOVSyncValue(iTime,timetype);
    userTime = (0==abtime);
    
    //std::cout<<"abtime "<<abtime<<std::endl;
    
    //query the IOVSequence
    cond::ValidityInterval validity = (*p).second->proxy()->setIntervalFor(abtime);
    
    recordValidity.first = std::max(recordValidity.first,validity.first);
    recordValidity.second = std::min(recordValidity.second,validity.second);
 
   //std::cout<<"setting validity "<<recordValidity.first<<" "<<recordValidity.second<<" for ibtime "<<abtime<< std::endl;
 
  }      
   
  // to force refresh we set end-value to the minimum such an IOV can exend to: current run or lumiblock
    
  if ( (!userTime) && recordValidity.second!=0) {
    edm::IOVSyncValue start = cond::toIOVSyncValue(recordValidity.first, timetype, true);
    edm::IOVSyncValue stop = doRefresh ? cond::limitedIOVSyncValue (iTime, timetype)
      : cond::toIOVSyncValue(recordValidity.second, timetype, false);
    
   
    oInterval = edm::ValidityInterval( start, stop );
   }
}
  

//required by EventSetup System
void 
PoolDBESSource::registerProxies(const edm::eventsetup::EventSetupRecordKey& iRecordKey , KeyedProxies& aProxyList) {
  std::string recordname=iRecordKey.name();

  ProxyMap::const_iterator b = m_proxies.lower_bound(recordname);
  ProxyMap::const_iterator e = m_proxies.upper_bound(recordname);
  if ( b == e) {
    LogDebug ("PoolDBESSource")<<"no DataProxy (Pluging) found for record "<<recordname;
    return;
  }

  for (ProxyMap::const_iterator p=b;p!=e;++p) {  
    if(0 != (*p).second.get()) {
      edm::eventsetup::TypeTag type =  (*p).second->type(); 
      edm::eventsetup::DataKey key( type, edm::eventsetup::IdTags((*p).second->label().c_str()) );
      aProxyList.push_back(KeyedProxies::value_type(key,(*p).second->edmProxy()));
    }
  }
}

// required by the EventSetup System
void 
PoolDBESSource::newInterval(const edm::eventsetup::EventSetupRecordKey& iRecordType,
			    const edm::ValidityInterval&) 
{
  //LogDebug ("PoolDBESSource")<<"newInterval";
  invalidateProxies(iRecordType);
}


// fills tagcollection merging with replacement
void 
PoolDBESSource::fillTagCollectionFromDB( cond::DbSession& coraldb, 
					 const std::string& roottag,
					 std::map<std::string,cond::TagMetadata>& replacement){
  //  std::cout<<"fillTagCollectionFromDB"<<std::endl;
  std::set< cond::TagMetadata > tagcoll;
  if (!roottag.empty()) {
    cond::TagCollectionRetriever tagRetriever( coraldb );
    tagRetriever.getTagCollection(roottag,tagcoll);
  } 

  std::set<cond::TagMetadata>::iterator it;
  std::set<cond::TagMetadata>::iterator itBeg=tagcoll.begin();
  std::set<cond::TagMetadata>::iterator itEnd=tagcoll.end();

  // FIXME the logic is a bit perverse: can be surely linearized (at least simplified!) ....
  for(it=itBeg; it!=itEnd; ++it){
    std::string k=it->recordname+"@"+it->labelname;
    std::map<std::string,cond::TagMetadata>::iterator fid=replacement.find(k);
    if(fid != replacement.end()){
      cond::TagMetadata m;
      m.recordname=it->recordname;
      m.labelname=it->labelname;
      m.pfn=fid->second.pfn;
      m.tag=fid->second.tag;
      m.objectname=it->objectname;
      m_tagCollection.insert(m);
      replacement.erase(fid);
    }else{
      m_tagCollection.insert(*it);
    }
  }
  std::map<std::string,cond::TagMetadata>::iterator itrep;
  std::map<std::string,cond::TagMetadata>::iterator itrepBeg=replacement.begin();
  std::map<std::string,cond::TagMetadata>::iterator itrepEnd=replacement.end();
  for(itrep=itrepBeg; itrep!=itrepEnd; ++itrep){
    //std::cout<<"appending"<<std::endl;
    //std::cout<<"pfn "<<itrep->second.pfn<<std::endl;
    //std::cout<<"objectname "<<itrep->second.objectname<<std::endl;
    //std::cout<<"tag "<<itrep->second.tag<<std::endl;
    //std::cout<<"recordname "<<itrep->second.recordname<<std::endl;
    m_tagCollection.insert(itrep->second);
  }
}
