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
#include <iterator>
#include <iostream>
#include <sstream>


namespace cond {

  namespace impl {
    struct FWMagic {
      // A.  Instantiate a plug-in manager first.
      edm::AssertHandler ah;
    };
  }

  FWIncantation::~FWIncantation(){}
  

  FWIncantation::FWIncantation() : magic(new impl::FWMagic) {

    // B.  Load the message service plug-in.  Forget this and bad things happen!
    //     In particular, the job hangs as soon as the output buffer fills up.
    //     That's because, without the message service, there is no mechanism for
    //     emptying the buffers.
    boost::shared_ptr<edm::Presence> theMessageServicePresence;
    theMessageServicePresence = boost::shared_ptr<edm::Presence>(edm::PresenceFactory::get()->
      makePresence("MessageServicePresence").release());

    // C.  Manufacture a configuration and establish it.
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
    
    
    boost::shared_ptr<std::vector<edm::ParameterSet> > pServiceSets;
    boost::shared_ptr<edm::ParameterSet>          params_;
    edm::makeParameterSets(config, params_, pServiceSets);
    
    // D.  Create the services.
    edm::ServiceToken tempToken(edm::ServiceRegistry::createSet(*pServiceSets.get()));
    
    // E.  Make the services available.
    edm::ServiceRegistry::Operate operate(tempToken);
    
  }

  //------------------------------------------------------------


  CondDB::CondDB() : me(0){}
  CondDB::CondDB(cond::Connection * conn) :
    me(conn) {
  }

  // move ownership....
  CondDB::CondDB(const CondDB & other) : me(other.me) {
    other.me=0;
  }

  CondDB & CondDB::operator=(const CondDB & other) {
    me = other.me;
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
  
  IOVProxy CondDB::iov(std::string const & tag) const {
    return IOVProxy(me->poolTransaction(),iovToken(tag),true);
  }
  
  IOVProxy CondDB::iovWithLib(std::string const & tag) const {
    return IOVProxy(me->poolTransaction(),iovToken(tag),false);
  }



  RDBMS::RDBMS() : session(new DBSession) {
    session->configuration().setAuthenticationMethod( cond::XML );
    session->configuration().setMessageLevel( cond::Error );
    session->open();
  }
  RDBMS::~RDBMS() {}

  RDBMS::RDBMS(std::string const & authPath) : session(new DBSession) {
    std::string authenv(std::string("CORAL_AUTH_PATH=")+authPath);
    ::putenv(const_cast<char*>(authenv.c_str()));
    session->configuration().setAuthenticationMethod( cond::XML );
    session->configuration().setMessageLevel( cond::Error );
    session->open();
  }
  
  RDBMS::RDBMS(std::string const & user,std::string const & pass) : session(new DBSession) {
    std::string userenv(std::string("CORAL_AUTH_USER=")+user);
    std::string passenv(std::string("CORAL_AUTH_PASSWORD=")+pass);
    ::putenv(const_cast<char*>(userenv.c_str()));
    ::putenv(const_cast<char*>(passenv.c_str()));
    session->configuration().setAuthenticationMethod( cond::Env );
    session->configuration().setMessageLevel( cond::Error );
    session->open();
  }
  
  CondDB RDBMS::getDB(std::string const & db) {
    cond::ConnectionHandler::Instance().registerConnection(db,*session,-1);
    cond::Connection & conn = *cond::ConnectionHandler::Instance().getConnection(db);
    conn.connect(session.get());
    return CondDB(&conn);
  }

}
