#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"

#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVProxy.h"

#include "CondCore/DBCommon/interface/Logger.h"
#include "CondCore/DBCommon/interface/LogDBEntry.h"
#include "CondCore/DBCommon/interface/TagInfo.h"

#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/IOVService/interface/IOVSchemaUtility.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/Utilities/interface/Utilities.h"
#include <iterator>
#include <limits>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cond {
  class ExportIOVUtilities : public Utilities {
    public:
      ExportIOVUtilities();
      ~ExportIOVUtilities();
      int execute();
  };
}

cond::ExportIOVUtilities::ExportIOVUtilities():Utilities("cmscond_export_iov"){
  addDictionaryOption();
  addAuthenticationOptions();
  addConfigFileOption();
  addLogDBOption();
  addConnectOption("sourceConnect","s","source connection string(required)");
  addConnectOption("destConnect","d","destionation connection string(required)");
  addOption<std::string>("inputTag","i","tag to export( default = destination tag)");
  addOption<std::string>("destTag","t","destination tag (required)");
  addOption<cond::Time_t>("beginTime","b","begin time (first since) (optional)");
  addOption<cond::Time_t>("endTime","e","end time (last till) (optional)");
  addOption<bool>("outOfOrder","o","allow out of order merge (optional, default=false)");
  addOption<bool>("exportMapping","m","export the mapping as in the source db (optional, default=false)");
  addOption<std::string>("usertext","x","user text, to be included in usertext column (optional, must be enclosed in double quotes)");
  addSQLOutputOption();
}

cond::ExportIOVUtilities::~ExportIOVUtilities(){
}

int cond::ExportIOVUtilities::execute(){
    
  std::string sourceConnect = getOptionValue<std::string>("sourceConnect");
  std::string destConnect = getOptionValue<std::string>("destConnect");

  std::string destTag = getOptionValue<std::string>("destTag");
  std::string inputTag(destTag);
  if( hasOptionValue("inputTag") ) inputTag = getOptionValue<std::string>("inputTag");
  std::string usertext("no user comments");
  if( hasOptionValue("usertext")) usertext = getOptionValue<std::string>("usertext");
  bool doLog = hasOptionValue("logDB");

  cond::Time_t since = std::numeric_limits<cond::Time_t>::min();
  if( hasOptionValue("beginTime" )) since = getOptionValue<cond::Time_t>("beginTime");
  cond::Time_t till = std::numeric_limits<cond::Time_t>::max();
  if( hasOptionValue("endTime" )) till = getOptionValue<cond::Time_t>("endTime");
  
  std::string sqlOutputFileName("sqlmonitoring.out");
  bool debug=hasDebug();
  bool outOfOrder = hasOptionValue("outOfOrder");

  std::string sourceiovtoken("");
  std::string destiovtoken("");
  bool newIOV = true;
  cond::TimeType sourceiovtype;

  cond::DbSession sourcedb = openDbSession("sourceConnect", true);
  cond::DbSession destdb = openDbSession("destConnect");
    
  std::auto_ptr<cond::Logger> logdb;
  cond::DbSession logSession;

  std::string payloadToken("");
  std::string payloadClasses("");
  int iovSize = 0;
  int ncopied = 0;
  cond::UserLogInfo a;
  if (doLog) {
    logSession = openDbSession( "logDB");
    logdb.reset(new cond::Logger(logSession));
    logdb->createLogDBIfNonExist();
    a.provenance=sourceConnect+"/"+inputTag;
    a.usertext="exportIOV V4.0;";
  }

  // find tag in source
  sourcedb.transaction().start(true);
  cond::MetaData  sourceMetadata(sourcedb);
  sourceiovtoken=sourceMetadata.getToken(inputTag);
  if(sourceiovtoken.empty()) 
    throw std::runtime_error(std::string("tag ")+inputTag+std::string(" not found") );
  
  if(debug){
    std::cout<<"source iov token "<<sourceiovtoken<<std::endl;
  }
  
  cond::IOVService iovmanager(sourcedb);
  sourceiovtype=iovmanager.timeType(sourceiovtoken);
  std::string const & timetypestr = cond::timeTypeSpecs[sourceiovtype].name;
  if(debug){
    std::cout<<"source iov type "<<sourceiovtype<<std::endl;
  }

  std::set<std::string> pclasses = iovmanager.payloadClasses( sourceiovtoken );
  
  if( doLog ){
    iovSize = iovmanager.iovSize( sourceiovtoken );
    std::ostringstream stream;
    std::copy(pclasses.begin(), pclasses.end(), std::ostream_iterator<std::string>(stream, ", "));
    payloadClasses = stream.str();
  }

  try{
    // find tag in destination
    cond::DbScopedTransaction transaction(destdb);
    transaction.start(false);

    cond::IOVSchemaUtility schemaUtil( destdb, std::cout );
    schemaUtil.createIOVContainerIfNecessary();

    destdb.storage().lockContainer( IOVNames::container() );
    int oldSize=0;
    cond::MetaData  metadata( destdb );
    if( metadata.hasTag(destTag) ){
      destiovtoken=metadata.getToken(destTag);
      newIOV = false;
      // grab info
      IOVProxy iov(destdb, destiovtoken);
      oldSize=iov.size();
      if (sourceiovtype!=iov.timetype()) {
	throw std::runtime_error("iov type in source and dest differs");
      }
    }
    if(debug){
      std::cout<<"destination iov token "<< destiovtoken <<std::endl;
    }
    
    since = std::max(since, cond::timeTypeSpecs[sourceiovtype].beginValue);
    till  = std::min(till,  cond::timeTypeSpecs[sourceiovtype].endValue);
    
    destiovtoken=iovmanager.exportIOVRangeWithPayload( destdb,
						       sourceiovtoken,
						       destiovtoken,
						       since, till,
						       outOfOrder );
    if (newIOV) {
      cond::MetaData destMetadata(destdb);
      destMetadata.addMapping(destTag,destiovtoken,sourceiovtype);
      if(debug){
	std::cout<<"dest iov token "<<destiovtoken<<std::endl;
	std::cout<<"dest iov type "<<sourceiovtype<<std::endl;
      }
      cond::IOVEditor iovEditor( destdb, destiovtoken );
      iovEditor.setScope( cond::IOVSequence::Tag );
    }
    
    
    ::sleep(1);
    
    // grab info
    // call IOV proxy with keep open option: it is required to lookup the payload class. A explicit commit will be needed at the end.
    if (doLog) {
      IOVProxy iov(destdb, destiovtoken);
      std::ostringstream stream;
      std::copy(iov.payloadClasses().begin(), iov.payloadClasses().end(), std::ostream_iterator<std::string>(stream, ", "));
      payloadClasses = stream.str();
      iovSize = iov.size();
      ncopied = iovSize-oldSize; 
      if ( ncopied == 1) {
	// get last object
	iov.tail(1);
	cond::IOVElementProxy last = *iov.begin();
	payloadToken=last.token();
	payloadClasses = destdb.classNameForItem( payloadToken );
      } 
      if (newIOV) a.usertext+= "new tag;";
      std::ostringstream ss;
      ss << "since="<< since <<", till="<< till << ", " << usertext << ";";
      ss << " copied="<< ncopied <<";";
      a.usertext +=ss.str();  
      logdb->logOperationNow(a,destConnect,payloadClasses,payloadToken,destTag,timetypestr,iovSize-1,since);
    }
    transaction.commit();
    sourcedb.transaction().commit();
  }catch ( cond::Exception const& er ){
    if (doLog) {
      if (newIOV) a.usertext+= "new tag;";
      std::ostringstream ss;
      ss << "since="<< since <<", till="<< till << ", " << usertext << ";";
      ss << " copied="<< ncopied <<";";
      a.usertext +=ss.str();
      logdb->logFailedOperationNow(a,destConnect,payloadClasses,payloadToken,destTag,timetypestr,iovSize-1,since,std::string(er.what()));
    }   
    sourcedb.transaction().commit();
    throw;
  }
      
  return 0;
}

int main( int argc, char** argv ){

  cond::ExportIOVUtilities utilities;
  return utilities.run(argc,argv);
}

