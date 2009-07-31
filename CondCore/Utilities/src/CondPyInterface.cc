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
#include "FWCore/ParameterSet/interface/MakeParameterSets.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"

#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/DBCommon/interface/Connection.h"
#include "CondCore/DBCommon/interface/AuthenticationMethod.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBCommon/interface/MessageLevel.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/FipProtocolParser.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVIterator.h"
#include "CondCore/DBCommon/interface/Logger.h"

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
    std::string config =
      "import FWCore.ParameterSet.Config as cms\n"
      "process = cms.Process('x')\n"
      "JobReportService = cms.Service('JobReportService')\n"
      "SiteLocalConfigService = cms.Service('SiteLocalConfigService')\n"
      ;

    boost::shared_ptr<std::vector<edm::ParameterSet> > pServiceSets;
    boost::shared_ptr<edm::ParameterSet>          params_;
    edm::makeParameterSets(config, params_, pServiceSets);
    
    // D.  Create the services.
    edm::ServiceToken tempToken(edm::ServiceRegistry::createSet(*pServiceSets.get()));
    
    // E.  Make the services available.
    magic->operate.reset(new edm::ServiceRegistry::Operate(tempToken));

  }

  //------------------------------------------------------------


  CondDB::CondDB() : me(0){
    //topinit();    
  }
  CondDB::CondDB(cond::Connection * conn, boost::shared_ptr<cond::Logger> ilog) :
    me(conn), logger(ilog) {
    //topinit();
  }

  // move ownership....
  CondDB::CondDB(const CondDB & other) : me(other.me), logger(other.logger) {
    other.me=0;
  }

  CondDB & CondDB::operator=(const CondDB & other) {
    if (me==other.me) return *this; // unless =0 this is an error condition!
    if (me!=0) me->disconnect();
    me = other.me;
    logger = other.logger;
    other.me=0;
    return *this;
  }

  CondDB::~CondDB() {
    if (me)
      me->disconnect();
  }
  

  std::string CondDB::allTags() const {
    std::ostringstream ss;

    cond::CoralTransaction& coraldb=me->coralTransaction();
    cond::MetaData metadata_svc(coraldb);
    std::vector<std::string> alltags;
    coraldb.start(true);
    metadata_svc.listAllTags(alltags);
    coraldb.commit();
    
    std::copy (alltags.begin(),
	       alltags.end(),
	       std::ostream_iterator<std::string>(ss," ")
	       );
    return ss.str();
  }

  std::string CondDB::iovToken(std::string const & tag) const {
    cond::CoralTransaction& coraldb=me->coralTransaction();
    cond::MetaData metadata_svc(coraldb);
    coraldb.start(true);
    std::string token=metadata_svc.getToken(tag);
    coraldb.commit();
    return token;
  }
  
  // fix commit problem....
  IOVProxy CondDB::iov(std::string const & tag) const {
    return IOVProxy(*me,iovToken(tag),true,true);
  }
  
  IOVProxy CondDB::iovWithLib(std::string const & tag) const {
    return IOVProxy(*me,iovToken(tag),false,true);
  }

  IOVElementProxy CondDB::payLoad(std::string const & token) const {
    return IOVElementProxy(0,0,token,me);

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



  RDBMS::RDBMS() : session(new DBSession) {
    //topinit();
    session->configuration().setAuthenticationMethod( cond::XML );
    session->configuration().setMessageLevel( cond::Error );
    session->configuration().setBlobStreamer( "COND/Services/TBufferBlobStreamingService" );
    session->open();
  }
  RDBMS::~RDBMS() {}

  RDBMS::RDBMS(std::string const & authPath,  bool debug) : session(new DBSession) {
    //topinit();
    session->configuration().setAuthenticationPath(authPath);
    session->configuration().setAuthenticationMethod( cond::XML );
    if (debug) session->configuration().setMessageLevel( cond::Debug );
    else
      session->configuration().setMessageLevel( cond::Error );
    session->configuration().setBlobStreamer( "COND/Services/TBufferBlobStreamingService" );
    session->open();
  }
  
  RDBMS::RDBMS(std::string const & user,std::string const & pass) : session(new DBSession) {
    //topinit();
    std::string userenv(std::string("CORAL_AUTH_USER=")+user);
    std::string passenv(std::string("CORAL_AUTH_PASSWORD=")+pass);
    ::putenv(const_cast<char*>(userenv.c_str()));
    ::putenv(const_cast<char*>(passenv.c_str()));
    session->configuration().setAuthenticationMethod( cond::Env );
    session->configuration().setMessageLevel( cond::Error );
    session->configuration().setBlobStreamer( "COND/Services/TBufferBlobStreamingService" );
    session->open();
  }

  void RDBMS::setLogger(std::string const & connstr) {
    cond::ConnectionHandler::Instance().registerConnection(connstr,*session,-1);
    cond::Connection & conn = *cond::ConnectionHandler::Instance().getConnection(connstr);
    conn.connect(session.get());
    logger.reset(new cond::Logger(&conn));
  }


  
  CondDB RDBMS::getDB(std::string const & db) {
    cond::ConnectionHandler::Instance().registerConnection(db,*session,-1);
    cond::Connection & conn = *cond::ConnectionHandler::Instance().getConnection(db);
    conn.connect(session.get());
    return CondDB(&conn,logger);
  }

}
