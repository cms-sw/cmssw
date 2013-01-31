#include "CondCore/Utilities/interface/CondPyInterface.h"

#include <exception>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/Utilities/interface/Presence.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
//#include "FWCore/PluginManager/interface/PluginCapabilities.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/FipProtocolParser.h"
#include "CondCore/DBCommon/interface/ClassID.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/DBCommon/interface/Logger.h"
#include "CondCore/TagCollection/interface/TagCollectionRetriever.h"

#include <iterator>
#include <iostream>
#include <sstream>


namespace cond {
  
  static void topinit(){
    if(!edmplugin::PluginManager::isAvailable())
      edmplugin::PluginManager::configure(edmplugin::standard::config());
    return;
  }

  namespace impl {
    struct FWMagic {
      // A.  Instantiate a plug-in manager first.
      edm::AssertHandler ah;
      boost::shared_ptr<edm::ServiceRegistry::Operate> operate;
    };
  }

  FWIncantation::~FWIncantation(){}
  

  FWIncantation::FWIncantation() : magic(new impl::FWMagic) {
    topinit();
    // B.  Load the message service plug-in.  Forget this and bad things happen!
    //     In particular, the job hangs as soon as the output buffer fills up.
    //     That's because, without the message service, there is no mechanism for
    //     emptying the buffers.
    boost::shared_ptr<edm::Presence> theMessageServicePresence;
    theMessageServicePresence = boost::shared_ptr<edm::Presence>(edm::PresenceFactory::get()->
								 makePresence("MessageServicePresence").release());
    
    // C.  Manufacture a configuration and establish it.

    /*
    std::string config =
      "process x = {"
      "service = MessageLogger {"
      "untracked vstring destinations = {'infos.mlog','warnings.mlog'}"
      "untracked PSet infos = {"
      "untracked string threshold = 'INFO'"
      "untracked PSet default = {untracked int32 limit = 1000000}"
      "untracked PSet FwkJob = {untracked int32 limit = 0}"
      "}"
      "untracked PSet warnings = {"
      "untracked string threshold = 'WARNING'"
      "untracked PSet default = {untracked int32 limit = 1000000}"
      "}"
      "untracked vstring fwkJobReports = {'FrameworkJobReport.xml'}"
      "untracked vstring categories = {'FwkJob'}"
      "untracked PSet FrameworkJobReport.xml = {"
      "untracked PSet default = {untracked int32 limit = 0}"
      "untracked PSet FwkJob = {untracked int32 limit = 10000000}"
      "}"
      "}"
      "service = JobReportService{}"
      "service = SiteLocalConfigService{}"
      "}";
    */
    /*
    std::string config =
      "import FWCore.ParameterSet.Config as cms\n"
      "process = cms.Process('x')\n"
      "JobReportService = cms.Service('JobReportService')\n"
      "SiteLocalConfigService = cms.Service('SiteLocalConfigService')\n"
      ;
    */
    
    boost::shared_ptr<std::vector<edm::ParameterSet> > psets(new std::vector<edm::ParameterSet>);
    edm::ParameterSet pSet;
    pSet.addParameter("@service_type",std::string("SiteLocalConfigService"));
    psets->push_back(pSet);
    
    // D.  Create the services.
    edm::ServiceToken tempToken(edm::ServiceRegistry::createSet(*psets.get()));
    
    // E.  Make the services available.
    magic->operate.reset(new edm::ServiceRegistry::Operate(tempToken));
    
  }
  
  //------------------------------------------------------------


  CondDB::CondDB() : me(){
    //topinit();    
  }
  CondDB::CondDB(cond::DbSession& session, boost::shared_ptr<cond::Logger> ilog) :
    me(session), logger(ilog) {
    //topinit();
  }

  // move ownership....
  CondDB::CondDB(const CondDB & other) : me(other.me), logger(other.logger) {
  }

  CondDB & CondDB::operator=(const CondDB & other) {
    if (this==&other) return *this; // unless =0 this is an error condition!
    me = other.me;
    logger = other.logger;
    return *this;
  }

  CondDB::~CondDB() {
  }
  

  std::string CondDB::allTags() const {
    std::ostringstream ss;

    cond::MetaData metadata_svc(me);
    std::vector<std::string> alltags;
    metadata_svc.listAllTags(alltags);
    
    std::copy (alltags.begin(),
	       alltags.end(),
	       std::ostream_iterator<std::string>(ss," ")
	       );
    return ss.str();
  }

  std::string CondDB::iovToken(std::string const & tag) const {
    cond::MetaData metadata_svc(me);
    std::string token=metadata_svc.getToken(tag);
    return token;
  }
  
  // fix commit problem....
  IOVProxy CondDB::iov(std::string const & tag) const {
    return IOVProxy(me,iovToken(tag));
  }
  
  IOVProxy CondDB::iovWithLib(std::string const & tag) const {
    return IOVProxy(me,iovToken(tag));
  }

  IOVElementProxy CondDB::payLoad(std::string const & token) const {
    ///FIXME: must be IOVElementProxy(since, till, token)
    return IOVElementProxy(0,0,token);
  }


  cond::LogDBEntry CondDB::lastLogEntry(std::string const & tag) const {
    cond::LogDBEntry entry;
    if (logger)
      logger->LookupLastEntryByTag(tag,entry,false);
    return entry;
  }

  cond::LogDBEntry CondDB::lastLogEntryOK(std::string const & tag) const{
    cond::LogDBEntry entry;
    if (logger)
      logger->LookupLastEntryByTag(tag,entry,true);
    return entry;
  }

  void CondDB::startTransaction() const {
    me.transaction().start();
  }
  void CondDB::startReadOnlyTransaction() const {
    me.transaction().start(true);
  }
  void CondDB::commitTransaction() const {
    me.transaction().commit();
  }

  void CondDB::closeSession() const {
    //FIXME: does the session disconnection fix the socket failure in FroNTier (bug #84265)?
    me.close();
  }
  
  RDBMS::RDBMS() : connection(new DbConnection) {
    //topinit();
    connection->configure( cond::CmsDefaults );
  }
  RDBMS::~RDBMS() {}

  RDBMS::RDBMS(std::string const & authPath,  bool debug) : connection(new DbConnection) {
    //topinit();
    connection->configuration().setAuthenticationPath(authPath);
    if (debug) 
      connection->configuration().setMessageLevel( coral::Debug );
    else
      connection->configuration().setMessageLevel( coral::Error );
    connection->configuration().setPoolAutomaticCleanUp( false );
    connection->configuration().setConnectionTimeOut(0);

    connection->configure();
  }
  
  RDBMS::RDBMS(std::string const & user,std::string const & pass) : connection(new DbConnection) {
    //topinit();
    std::string userenv(std::string("CORAL_AUTH_USER=")+user);
    std::string passenv(std::string("CORAL_AUTH_PASSWORD=")+pass);
    ::putenv(const_cast<char*>(userenv.c_str()));
    ::putenv(const_cast<char*>(passenv.c_str()));
    connection->configuration().setMessageLevel( coral::Error );
    connection->configuration().setPoolAutomaticCleanUp( false );
    connection->configuration().setConnectionTimeOut(0);
    connection->configure();
  }

  void RDBMS::setLogger(std::string const & connstr) {
    DbSession loggerSession = connection->createSession();
    logger.reset(new cond::Logger(loggerSession));
    logger->connect( connstr, true );
  }


  
  CondDB RDBMS::getDB(std::string const & db) {
    DbSession dbSession = connection->createSession();
    dbSession.open( db );
    return CondDB(dbSession,logger);
  }

  CondDB RDBMS::getReadOnlyDB(std::string const & db) {
    DbSession dbSession = connection->createSession();
    dbSession.open( db, true );
    return CondDB(dbSession,logger);
  }
  
  GlobalTag const &  RDBMS::globalTag(std::string const & connstr, 
				      std::string const & gname,
				      std::string const & prefix, 
				      std::string const & postfix) const {
    DbSession session = connection->createSession();
    session.open( connstr, true );
    session.transaction().start( true );
    TagCollectionRetriever gtr(session, prefix,postfix);
    const_cast<GlobalTag&>(m_globalTag).clear();
    gtr.getTagCollection(gname,const_cast<GlobalTag&>(m_globalTag));
    session.transaction().commit();
    return m_globalTag;  
  }


}
