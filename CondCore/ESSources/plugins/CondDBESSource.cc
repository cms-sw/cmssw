//
// Package:     CondCore/ESSources
// Module:      CondDBESSource
//
// Description: <one line class summary>
//
// Implementation:
//     <Notes on implementation>
//
// Author:      Zhen Xie
//
#include "CondDBESSource.h"

#include "boost/shared_ptr.hpp"
#include <boost/algorithm/string.hpp>
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Auth.h"
#include "CondFormats/Common/interface/Time.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/ConvertIOVSyncValue.h"

#include "CondCore/ESSources/interface/ProxyFactory.h"
#include "CondCore/ESSources/interface/DataProxy.h"

#include "CondCore/IOVService/interface/PayloadProxy.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <exception>
//#include <cstdlib>
//#include <iostream>


#include "CondFormats/Common/interface/TimeConversions.h"
#include <iomanip>

namespace {
  /* utility ot build the name of the plugin corresponding to a given record
     se ESSources
   */
  std::string
  buildName( std::string const & iRecordName ) {
    return iRecordName + std::string( "@NewProxy" );
  }
  
  std::string joinRecordAndLabel( std::string const & iRecordName, std::string const & iLabelName ) {
    return iRecordName + std::string( "@" ) + iLabelName;
  }

  /* utility class to return a IOVs associated to a given "name"
     This implementation return the IOV associated to a record...
     It is essentialy a workaround to get the full IOV out of the tag colector
     that is not accessible as hidden in the ESSource
     FIXME: need to support label??
   */
  class CondGetterFromESSource : public cond::CondGetter {
  public:
    CondGetterFromESSource(CondDBESSource::ProxyMap const & ip) : m_proxies(ip){}
    virtual ~CondGetterFromESSource(){}

    cond::IOVProxy get(std::string name) const {
      CondDBESSource::ProxyMap::const_iterator p = m_proxies.find(name);
      if ( p != m_proxies.end())
	return (*p).second->proxy()->iov();
      return cond::IOVProxy();
    }

    CondDBESSource::ProxyMap const & m_proxies;
  };

  // dump the state of a DataProxy
  void dumpInfo(std::ostream & out, std::string const & recName, cond::DataProxyWrapperBase const & proxy) {
    cond::SequenceState state(proxy.proxy()->iov().state());
    out << recName << " / " << proxy.label() << ": " 
	<< proxy.connString() << ", " << proxy.tag()   << "\n  "
	<< state.size() << ", " << state.revision()  << ", "
	<< cond::time::to_boost(state.timestamp())     << "\n  "
	<< state.comment()
	<< "\n  "
	<< "refresh " << proxy.proxy()->stats.nRefresh
	<< "/" << proxy.proxy()->stats.nArefresh
	<< ", reconnect " << proxy.proxy()->stats.nReconnect
	<< "/" << proxy.proxy()->stats.nAreconnect
	<< ", make " << proxy.proxy()->stats.nMake
	<< ", load " << proxy.proxy()->stats.nLoad
      ;
    if ( proxy.proxy()->stats.nLoad>0) {
      out << "\n oids,sinces:";
      cond::BasePayloadProxy::ObjIds const & ids =  proxy.proxy()->stats.ids;
      for (cond::BasePayloadProxy::ObjIds::const_iterator id=ids.begin(); id!=ids.end(); ++id)
	out << " "
	    // << std::ios::hex 
            << (*id).oid1 <<"-"<< (*id).oid2 <<"," 
	    // << std::ios::dec 
            <<  (*id).since;
    }
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
CondDBESSource::CondDBESSource( const edm::ParameterSet& iConfig ) :
  m_connection(), 
  m_lastRun(0),  // for the stat
  m_lastLumi(0),  // for the stat
  m_policy( NOREFRESH ),
  m_doDump( iConfig.getUntrackedParameter<bool>( "DumpStat", false ) )
{
  if( iConfig.getUntrackedParameter<bool>( "RefreshAlways", false ) ) {
    m_policy = REFRESH_ALWAYS;
  }
  if( iConfig.getUntrackedParameter<bool>( "RefreshOpenIOVs", false ) ) {
    m_policy = REFRESH_OPEN_IOVS;
  }
  if( iConfig.getUntrackedParameter<bool>( "RefreshEachRun", false ) ) {
    m_policy = REFRESH_EACH_RUN;
  }
  if( iConfig.getUntrackedParameter<bool>( "ReconnectEachRun", false ) ) {
    m_policy = RECONNECT_EACH_RUN;
  }

  Stats s = {0,0,0,0,0,0,0,0};
  m_stats = s;	
  //std::cout<<"CondDBESSource::CondDBESSource"<<std::endl;
  /*parameter set parsing and pool environment setting
   */
  
  // default connection string
  // inproduction used for the global tag
  std::string userconnect= iConfig.getParameter<std::string>("connect");
  

  // connection configuration
  edm::ParameterSet connectionPset = iConfig.getParameter<edm::ParameterSet>( "DBParameters" );
  m_connection.configuration().setParameters( connectionPset );
  m_connection.configure();
  

  // load additional record/tag info it will overwrite the global tag
  std::map<std::string,cond::TagMetadata> replacement;
  if( iConfig.exists( "toGet" ) ) {
    typedef std::vector< edm::ParameterSet > Parameters;
    Parameters toGet = iConfig.getParameter<Parameters>( "toGet" );
    for( Parameters::iterator itToGet = toGet.begin(); itToGet != toGet.end(); ++itToGet ) {
      cond::TagMetadata nm;
      nm.recordname = itToGet->getParameter<std::string>( "record" );
      nm.labelname = itToGet->getUntrackedParameter<std::string>( "label", "" );
      nm.tag = itToGet->getParameter<std::string>( "tag" );
      nm.pfn = itToGet->getUntrackedParameter<std::string>( "connect", userconnect );
      //	nm.objectname=itFound->second;
      std::string recordLabelKey = joinRecordAndLabel( nm.recordname, nm.labelname );
      replacement.insert( std::pair<std::string,cond::TagMetadata>( recordLabelKey, nm ) );
    }
  }
  
  // get the global tag, merge with "replacement" store in "tagCollection"
  std::vector<std::string> globaltagList;
  std::vector<std::string> connectList;
  std::vector<std::string> pfnPrefixList;
  std::vector<std::string> pfnPostfixList;
  if( iConfig.exists( "globaltag" ) ) {
    std::string pfnPrefix(iConfig.getUntrackedParameter<std::string>( "pfnPrefix", "" ));
    std::string pfnPostfix(iConfig.getUntrackedParameter<std::string>( "pfnPostfix", "" ));
    std::string globaltag(iConfig.getParameter<std::string>( "globaltag" ));
    boost::split( globaltagList, globaltag, boost::is_any_of("|"), boost::token_compress_off );
    fillList(userconnect, connectList, globaltagList.size(), "connection");
    fillList(pfnPrefix, pfnPrefixList, globaltagList.size(), "pfnPrefix");
    fillList(pfnPostfix, pfnPostfixList, globaltagList.size(), "pfnPostfix");
  }

  fillTagCollectionFromDB(connectList,
			  pfnPrefixList,
			  pfnPostfixList,
			  globaltagList,
			  replacement);
  
  TagCollection::iterator it;
  TagCollection::iterator itBeg = m_tagCollection.begin();
  TagCollection::iterator itEnd = m_tagCollection.end();
 
  std::map<std::string, cond::DbSession> sessions;

  /* load DataProxy Plugin (it is strongly typed due to EventSetup ideosyncrasis)
   * construct proxy
   * contrary to EventSetup the "object-name" is not used as identifier: multiple entries in a record are
   * dinstinguished only by their label...
   * done in two step: first create ProxyWrapper loading ALL required dictionaries
   * this will allow to initialize POOL in one go for each "database"
   * The real initialization of the Data-Proxies is done in the second loop 
   */
  std::vector<cond::DataProxyWrapperBase *> proxyWrappers(m_tagCollection.size());
  size_t ipb=0;
  for(it=itBeg;it!=itEnd;++it){
    proxyWrappers[ipb++] =  
      cond::ProxyFactory::get()->create(buildName(it->second.recordname));
  }
  // now all required libraries have been loaded
  // init sessions and DataProxies
  ipb=0;
  for(it=itBeg;it!=itEnd;++it){
    std::map<std::string, cond::DbSession>::iterator p = sessions.find( it->second.pfn );
    cond::DbSession nsess;
    if (p==sessions.end()) {
      //open db get tag info (i.e. the IOV token...)
      nsess = m_connection.createSession();
      nsess.openReadOnly( it->second.pfn, "" );
      sessions.insert(std::make_pair(it->second.pfn,nsess));
    } else nsess = (*p).second;
    //cond::MetaData metadata(nsess);
    //nsess.transaction().start(true);
    //std::string iovtoken = metadata.getToken(it->tag);
    // owenship...
    ProxyP proxy(proxyWrappers[ipb++]);
   //  instert in the map
    m_proxies.insert(std::make_pair(it->second.recordname, proxy));
    // initialize
    //proxy->lateInit(nsess,iovtoken, 
    //		    it->labelname, it->pfn, it->tag
    //	    );
    proxy->lateInit(nsess,it->second.tag, 
		    it->second.labelname, it->second.pfn);
    //nsess.transaction().commit();
  }

  // one loaded expose all other tags to the Proxy! 
  CondGetterFromESSource visitor( m_proxies );
  ProxyMap::iterator b = m_proxies.begin();
  ProxyMap::iterator e = m_proxies.end();
  for ( ;b != e; b++ ) {

    (*b).second->proxy()->loadMore( visitor );

    /// required by eventsetup
    edm::eventsetup::EventSetupRecordKey recordKey(edm::eventsetup::EventSetupRecordKey::TypeTag::findType( (*b).first ) );
    if( recordKey.type() != edm::eventsetup::EventSetupRecordKey::TypeTag() ) {
      findingRecordWithKey( recordKey );
      usingRecordWithKey( recordKey );
    }
  }

  m_stats.nData=m_proxies.size();

}

void CondDBESSource::fillList(const std::string & stringList, std::vector<std::string> & listToFill, const unsigned int listSize, const std::string & type)
{
  boost::split( listToFill, stringList, boost::is_any_of("|"), boost::token_compress_off );
  // If it is one clone it for each GT
  if( listToFill.size() == 1 ) {
    for( unsigned int i=1; i<listSize; ++i ) {
      listToFill.push_back(stringList);
    }
  }
  // else if they don't match the number of GTs throw an exception
  else if( listSize != listToFill.size() ) {
    throw cond::Exception( std::string( "ESSource: number of global tag components does not match number of "+type+" strings" ) );
  }
}

CondDBESSource::~CondDBESSource() {
  //dump info FIXME: find a more suitable place...
  if (m_doDump) {
    std::cout << "CondDBESSource Statistics" << std::endl
	      << "DataProxy " << m_stats.nData
	      << " setInterval " << m_stats.nSet
	      << " Runs " << m_stats.nRun
	      << " Lumis " << m_stats.nLumi
	      << " Refresh " << m_stats.nRefresh
	      << " Actual Refresh " << m_stats.nActualRefresh
	      << " Reconnect " << m_stats.nReconnect
	      << " Actual Reconnect " << m_stats.nActualReconnect;
    std::cout << std::endl;
    std::cout << "Global Proxy Statistics" << std::endl
	      << "proxy " << cond::BasePayloadProxy::gstats.nProxy
	      << " make " << cond::BasePayloadProxy::gstats.nMake
	      << " load " << cond::BasePayloadProxy::gstats.nLoad;
    std::cout << std::endl;


    ProxyMap::iterator b= m_proxies.begin();
    ProxyMap::iterator e= m_proxies.end();
    for ( ;b != e; b++ ) {
      dumpInfo( std::cout, (*b).first, *(*b).second );
      std::cout << "\n" << std::endl;
    }

    // FIXME
    // We shall eventually close transaction and session...
  }
}


//
// invoked by EventSetUp: for a given record return the smallest IOV for which iTime is valid
// limit to next run/lumisection of Refresh is required
//
void 
CondDBESSource::setIntervalFor( const edm::eventsetup::EventSetupRecordKey& iKey, const edm::IOVSyncValue& iTime, edm::ValidityInterval& oInterval ){

  std::string recordname=iKey.name();
  
  edm::LogInfo( "CondDBESSource" ) << "Getting data for record \""<< recordname
				   << "\" to be consumed by "<< iTime.eventID() << ", timestamp: " << iTime.time().value()
				   << "; from CondDBESSource::setIntervalFor";
  
  m_stats.nSet++;
  //{
    // not really required, keep here for the time being
    if(iTime.eventID().run()!=m_lastRun) {
      m_lastRun=iTime.eventID().run();
      m_stats.nRun++;
    }
    if(iTime.luminosityBlockNumber()!=m_lastLumi) {
      m_lastLumi=iTime.luminosityBlockNumber();
      m_stats.nLumi++;
    }
    //}
 
  bool doRefresh = false;
  if( m_policy == REFRESH_EACH_RUN || m_policy == RECONNECT_EACH_RUN ) {
    // find out the last run number for the proxy of the specified record
    std::map<std::string,unsigned int>::iterator iRec = m_lastRecordRuns.find( recordname );
    if( iRec != m_lastRecordRuns.end() ){
      unsigned int lastRecordRun = iRec->second;
      if( lastRecordRun != m_lastRun ){
        // a refresh is required!
        doRefresh = true;
        iRec->second = m_lastRun;
	edm::LogInfo( "CondDBESSource" ) << "Preparing refresh for record \"" << recordname 
					 << "\" since there has been a transition from run "
					 << lastRecordRun << " to run " << m_lastRun
					 << "; from CondDBESSource::setIntervalFor";
      }
    } else {
      doRefresh = true;
      m_lastRecordRuns.insert( std::make_pair( recordname, m_lastRun ) );
      edm::LogInfo( "CondDBESSource" ) << "Preparing refresh for record \"" << recordname 
				       << "\" for " << iTime.eventID() << ", timestamp: " << iTime.time().value()
				       << "; from CondDBESSource::setIntervalFor";
    }
    if ( !doRefresh )
      edm::LogInfo( "CondDBESSource" ) << "Though enabled, refresh not needed for record \"" << recordname 
				       << "\" for " << iTime.eventID() << ", timestamp: " << iTime.time().value()
				       << "; from CondDBESSource::setIntervalFor";
  } else if( m_policy == REFRESH_ALWAYS || m_policy == REFRESH_OPEN_IOVS ) {
    doRefresh = true;
    edm::LogInfo( "CondDBESSource" ) << "Forcing refresh for record \"" << recordname 
				     << "\" for " << iTime.eventID() << ", timestamp: " << iTime.time().value()
				     << "; from CondDBESSource::setIntervalFor";
  }

  oInterval = edm::ValidityInterval::invalidInterval();

  // compute the smallest interval (assume all objects have the same timetype....)                                                                                                          
  cond::ValidityInterval recordValidity(1,cond::TIMELIMIT);
  cond::TimeType timetype;
  bool userTime=true;

 //FIXME use equal_range
  ProxyMap::const_iterator pmBegin = m_proxies.lower_bound(recordname);
  ProxyMap::const_iterator pmEnd = m_proxies.upper_bound(recordname);
  if ( pmBegin == pmEnd ) {
    edm::LogInfo( "CondDBESSource" ) << "No DataProxy (Pluging) found for record \""<< recordname
				     << "\"; from CondDBESSource::setIntervalFor";
    return;
  }
  
  for ( ProxyMap::const_iterator pmIter = pmBegin; pmIter != pmEnd; ++pmIter ) {

    edm::LogInfo( "CondDBESSource" ) << "Processing record \"" << recordname
				     << "\" and label \""<< pmIter->second->label()
				     << "\" for " << iTime.eventID() << ", timestamp: " << iTime.time().value()
				     << "; from CondDBESSource::setIntervalFor";

    timetype = (*pmIter).second->proxy()->timetype();
    
    cond::Time_t abtime = cond::fromIOVSyncValue( iTime, timetype );
    userTime = ( 0 == abtime );
    
    //std::cout<<"abtime "<<abtime<<std::endl;

    if (userTime) return; //  oInterval invalid to avoid that make is called...


    
    if( doRefresh ) {

      std::string recKey = joinRecordAndLabel( recordname, pmIter->second->label() );
      TagCollection::const_iterator tcIter = m_tagCollection.find( recKey ); 
      if ( tcIter == m_tagCollection.end() ) {
	edm::LogInfo( "CondDBESSource" ) << "No Tag found for record \""<< recordname
					 << "\" and label \""<< pmIter->second->label()
					 << "\"; from CondDBESSource::setIntervalFor";
	return;
      }

      // first reconnect if required
      if( m_policy == RECONNECT_EACH_RUN ) {
	edm::LogInfo( "CondDBESSource" ) << "Checking if the session must be closed and re-opened for getting correct conditions"
					 << "; from CondDBESSource::setIntervalFor";
	std::stringstream transId;
	//transId << "long" << m_lastRun;
	transId << m_lastRun;
	std::map<std::string,std::pair<cond::DbSession,std::string> >::iterator iSess = m_sessionPool.find( tcIter->second.pfn );
	cond::DbSession theSession;
	bool reopen = false;
	if( iSess != m_sessionPool.end() ){
	  if( iSess->second.second != transId.str() ) {
	    // the available session is open for a different run: reopen
            reopen = true;
	    iSess->second.second = transId.str();
	  }
	  theSession = iSess->second.first;
	} else {
          // no available session: probably first run analysed... 
	  theSession = m_connection.createSession(); 
	  m_sessionPool.insert(std::make_pair( tcIter->second.pfn,std::make_pair(theSession,transId.str()) )); 
	  reopen = true;
	} 
	if( reopen ){
	  theSession.openReadOnly( tcIter->second.pfn, transId.str() );
	  edm::LogInfo( "CondDBESSource" ) << "Re-opening the session with connection string " << tcIter->second.pfn
					   << " and new transaction Id " <<  transId.str()
					   << "; from CondDBESSource::setIntervalFor";
	}
	
	edm::LogInfo( "CondDBESSource" ) << "Reconnecting to \"" << tcIter->second.pfn
					 << "\" for getting new payload for record \"" << recordname 
					 << "\" and label \""<< pmIter->second->label()
					 << "\" from IOV tag \"" << tcIter->second.tag
					 << "\" to be consumed by " << iTime.eventID() << ", timestamp: " << iTime.time().value()
					 << "; from CondDBESSource::setIntervalFor";
	bool isSizeIncreased = pmIter->second->proxy()->refresh( theSession );
	if( isSizeIncreased )
	  edm::LogInfo( "CondDBESSource" ) << "After reconnecting, an increased size of the IOV sequence labeled by tag \"" << tcIter->second.tag
					   << "\" was found; from CondDBESSource::setIntervalFor";
	m_stats.nActualReconnect += isSizeIncreased;
	m_stats.nReconnect++;
      } else {
	edm::LogInfo( "CondDBESSource" ) << "Refreshing IOV sequence labeled by tag \"" << tcIter->second.tag
					 << "\" for getting new payload for record \"" << recordname
					 << "\" and label \""<< pmIter->second->label()
					 << "\" to be consumed by " << iTime.eventID() << ", timestamp: " << iTime.time().value()
					 << "; from CondDBESSource::setIntervalFor";
	bool isSizeIncreased = pmIter->second->proxy()->refresh();
	if( isSizeIncreased )
	  edm::LogInfo( "CondDBESSource" ) << "After refreshing, an increased size of the IOV sequence labeled by tag \"" << tcIter->second.tag
					   << "\" was found; from CondDBESSource::setIntervalFor";
        m_stats.nActualRefresh += isSizeIncreased;
	m_stats.nRefresh++;
      }

    }

    /*
      // make oInterval valid For Ever
    {
     oInterval = edm::ValidityInterval(cond::toIOVSyncValue(recordValidity.first,  cond::runnumber, true), 
                                       cond::toIOVSyncValue(recordValidity.second, cond::runnumber, false));
     return;
    }    
    */

    //query the IOVSequence
    cond::ValidityInterval validity = (*pmIter).second->proxy()->setIntervalFor( abtime );
    
    edm::LogInfo( "CondDBESSource" ) << "Validity coming from IOV sequence for record \"" << recordname
				     << "\" and label \""<< pmIter->second->label()
				     << "\": (" << validity.first << ", " << validity.second
				     << ") for time (type: "<< cond::timeTypeNames( timetype ) << ") " << abtime
				     << "; from CondDBESSource::setIntervalFor";
    
    recordValidity.first = std::max(recordValidity.first,validity.first);
    recordValidity.second = std::min(recordValidity.second,validity.second);
  }      
  
  if( m_policy == REFRESH_OPEN_IOVS ) {
    doRefresh = ( recordValidity.second == cond::timeTypeSpecs[timetype].endValue );
    edm::LogInfo( "CondDBESSource" ) << "Validity for record \"" << recordname
				     << "\" and the corresponding label(s) coming from Condition DB: (" << recordValidity.first 
				     << ", "<< recordValidity.first 
				     << ") as the last IOV element in the IOV sequence is infinity"
				     << "; from CondDBESSource::setIntervalFor";
  }
  
  // to force refresh we set end-value to the minimum such an IOV can extend to: current run or lumiblock
    
  if ( (!userTime) && recordValidity.second !=0 ) {
    edm::IOVSyncValue start = cond::toIOVSyncValue(recordValidity.first, timetype, true);
    edm::IOVSyncValue stop = doRefresh  ? cond::limitedIOVSyncValue (iTime, timetype)
      : cond::toIOVSyncValue(recordValidity.second, timetype, false);
       
    oInterval = edm::ValidityInterval( start, stop );
   }
  
  edm::LogInfo( "CondDBESSource" ) << "Setting validity for record \"" << recordname 
				   << "\" and corresponding label(s): starting at " << oInterval.first().eventID() << ", timestamp: " << oInterval.first().time().value()
				   << ", ending at "<< oInterval.last().eventID() << ", timestamp: " << oInterval.last().time().value()
				   << ", for "<< iTime.eventID() << ", timestamp: " << iTime.time().value()
				   << "; from CondDBESSource::setIntervalFor";
}
  

//required by EventSetup System
void 
CondDBESSource::registerProxies(const edm::eventsetup::EventSetupRecordKey& iRecordKey , KeyedProxies& aProxyList) {
  std::string recordname=iRecordKey.name();

  ProxyMap::const_iterator b = m_proxies.lower_bound(recordname);
  ProxyMap::const_iterator e = m_proxies.upper_bound(recordname);
  if ( b == e) {
    edm::LogInfo( "CondDBESSource" ) << "No DataProxy (Pluging) found for record \""<< recordname
				     << "\"; from CondDBESSource::registerProxies";
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
CondDBESSource::newInterval(const edm::eventsetup::EventSetupRecordKey& iRecordType,
			    const edm::ValidityInterval&) 
{
  //LogDebug ("CondDBESSource")<<"newInterval";
  invalidateProxies(iRecordType);
}


// Fills tag collection from the given globaltag
void CondDBESSource::fillTagCollectionFromGT( const std::string & coraldb,
                                              const std::string & prefix,
                                              const std::string & postfix,
                                              const std::string & roottag,
                                              std::set< cond::TagMetadata > & tagcoll )
{
  // std::cout << "coraldb = " << coraldb << std::endl;
  // std::cout << "prefix = " << prefix << std::endl;
  // std::cout << "postfix = " << postfix << std::endl;
  // std::cout << "roottag = " << roottag << std::endl;
  if ( !roottag.empty() ) {
    if ( coraldb.empty() )
      throw cond::Exception( std::string( "ESSource: requested global tag ") + roottag + std::string( " but not connection string given" ) );
    cond::DbSession session = m_connection.createSession();
    session.open( coraldb, cond::Auth::COND_READER_ROLE, true );
    session.transaction().start( true );
    cond::TagCollectionRetriever tagRetriever( session, prefix, postfix );
    tagRetriever.getTagCollection( roottag,tagcoll );
    session.transaction().commit();
  }
}

// fills tagcollection merging with replacement
// Note: it assumem the coraldbList and roottagList have the same length. This checked in the constructor that prepares the two lists before calling this method.
void CondDBESSource::fillTagCollectionFromDB( const std::vector<std::string> & coraldbList,
                                              const std::vector<std::string> & prefixList,
                                              const std::vector<std::string> & postfixList,
                                              const std::vector<std::string> & roottagList,
                                              std::map<std::string,cond::TagMetadata>& replacement )
{
  std::set< cond::TagMetadata > tagcoll;
 
  auto coraldb = coraldbList.begin();
  auto prefix = prefixList.begin();
  auto postfix = postfixList.begin();
  for( auto roottag = roottagList.begin(); roottag != roottagList.end(); ++roottag, ++coraldb, ++prefix, ++postfix) {
    fillTagCollectionFromGT(*coraldb, *prefix, *postfix, *roottag, tagcoll);
  }

  std::set<cond::TagMetadata>::iterator tagCollIter;
  std::set<cond::TagMetadata>::iterator tagCollBegin = tagcoll.begin();
  std::set<cond::TagMetadata>::iterator tagCollEnd = tagcoll.end();

  // FIXME the logic is a bit perverse: can be surely linearized (at least simplified!) ....
  for( tagCollIter = tagCollBegin; tagCollIter != tagCollEnd; ++tagCollIter ) {
    std::string recordLabelKey = joinRecordAndLabel( tagCollIter->recordname, tagCollIter->labelname );
    std::map<std::string,cond::TagMetadata>::iterator fid = replacement.find( recordLabelKey );
    if( fid != replacement.end() ) {
      cond::TagMetadata tagMetadata;
      tagMetadata.recordname = tagCollIter->recordname;
      tagMetadata.labelname = tagCollIter->labelname;
      tagMetadata.pfn = fid->second.pfn;
      tagMetadata.tag = fid->second.tag;
      tagMetadata.objectname = tagCollIter->objectname;
      m_tagCollection.insert( std::make_pair( recordLabelKey, tagMetadata ) );
      replacement.erase( fid );
      edm::LogInfo( "CondDBESSource" ) << "Replacing connection string \"" << tagCollIter->pfn
				       << "\" and tag \"" << tagCollIter->tag
				       << "\" for record \"" << tagMetadata.recordname
				       << "\" and label \"" << tagMetadata.labelname
				       << "\" with connection string \"" << tagMetadata.pfn
				       << "\" and tag " << tagMetadata.tag
				       << "\"; from CondDBESSource::fillTagCollectionFromDB";
    } else {
      m_tagCollection.insert( std::make_pair( recordLabelKey, *tagCollIter) );
    }
  }
  std::map<std::string,cond::TagMetadata>::iterator replacementIter;
  std::map<std::string,cond::TagMetadata>::iterator replacementBegin = replacement.begin();
  std::map<std::string,cond::TagMetadata>::iterator replacementEnd = replacement.end();
  for( replacementIter = replacementBegin; replacementIter != replacementEnd; ++replacementIter ){
    // std::cout<<"appending"<<std::endl;
    // std::cout<<"pfn "<<replacementIter->second.pfn<<std::endl;
    // std::cout<<"objectname "<<replacementIter->second.objectname<<std::endl;
    // std::cout<<"tag "<<replacementIter->second.tag<<std::endl;
    // std::cout<<"recordname "<<replacementIter->second.recordname<<std::endl;
    m_tagCollection.insert( *replacementIter );
  }
}


// backward compatibility for configuration files
class PoolDBESSource : public CondDBESSource {
public:
  explicit  PoolDBESSource( const edm::ParameterSet& ps) :
    CondDBESSource(ps){}
};

#include "FWCore/Framework/interface/SourceFactory.h"
//define this as a plug-in
DEFINE_FWK_EVENTSETUP_SOURCE(PoolDBESSource);

