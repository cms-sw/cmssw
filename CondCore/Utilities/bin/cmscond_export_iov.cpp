#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"

#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/IOVService/interface/IOVIterator.h"

#include "CondCore/DBCommon/interface/Logger.h"
#include "CondCore/DBCommon/interface/LogDBEntry.h"
#include "CondCore/DBCommon/interface/UserLogInfo.h"
#include "CondCore/DBCommon/interface/TagInfo.h"

#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/Utilities/interface/Utilities.h"
#include <iterator>
#include <limits>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include<sstream>

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
  addBlobStreamerOption();
  addLogDBOption();
  addConnectOption("sourceConnect","s","source connection string(required)");
  addConnectOption("destConnect","d","destionation connection string(required)");
  addOption<std::string>("inputTag","i","tag to export( default = destination tag)");
  addOption<std::string>("destTag","t","destination tag (required)");
  addOption<cond::Time_t>("beginTime","b","begin time (first since) (optional)");
  addOption<cond::Time_t>("endTime","e","end time (last till) (optional)");
  addOption<bool>("outOfOrder","o","allow out of order merge (optional, default=false)");
  addOption<std::string>("usertext","x","user text, to be included in usertext column (optional, must be enclosed in double quotes)");
  addSQLOutputOption();
}

cond::ExportIOVUtilities::~ExportIOVUtilities(){
}

int cond::ExportIOVUtilities::execute(){
    
  std::string sourceConnect = getOptionValue<std::string>("sourceConnect");
  std::string destConnect = getOptionValue<std::string>("destConnect");
  
  std::string destTag = getOptionValue<std::string>("destConnect");
  std::string inputTag("");
  if( hasOptionValue("inputTag") ) inputTag = getOptionValue<std::string>("inputTag");
  std::string usertext("no user comments");
  if( hasOptionValue("usertext")) usertext = getOptionValue<std::string>("usertext");
  std::string logConnect = getOptionValue<std::string>("logDB");

  cond::Time_t since = std::numeric_limits<cond::Time_t>::min();
  if( hasOptionValue("beginTime" )) since = getOptionValue<cond::Time_t>("beginTime");
  cond::Time_t till = std::numeric_limits<cond::Time_t>::max();
  if( hasOptionValue("endTime" )) till = getOptionValue<cond::Time_t>("endTime");
  
  std::string sqlOutputFileName("sqlmonitoring.out");
  bool doLog = hasOptionValue("logDB");
  bool debug=hasDebug();
  std::string blobStreamerName("COND/Services/TBufferBlobStreamingService2");
  bool outOfOrder = hasOptionValue("outOfOrder");

  std::string sourceiovtoken("");
  std::string destiovtoken("");
  cond::TimeType sourceiovtype;

  cond::DbSession sourcedb = openDbSession("sourceConnect");
  cond::DbSession destdb = openDbSession("destConnect");
    
  // find tag in source
  {
    sourcedb.transaction().start(true);
    cond::MetaData  sourceMetadata(sourcedb);
    if( !sourceMetadata.hasTag(inputTag) ){
      throw std::runtime_error(std::string("tag ")+inputTag+std::string(" not found") );
    }
    //sourceiovtoken=sourceMetadata->getToken(inputTag);
    cond::MetaDataEntry entry;
    sourceMetadata.getEntryByTag(inputTag,entry);
    sourceiovtoken=entry.iovtoken;
    sourceiovtype=entry.timetype;

    sourcedb.transaction().commit();
    if(debug){
      std::cout<<"source iov token "<<sourceiovtoken<<std::endl;
      std::cout<<"source iov type "<<sourceiovtype<<std::endl;
    }
  }
  // find tag in destination
  {
    cond::DbScopedTransaction transaction(destdb);
    transaction.start(false);
    destdb.initializeMapping( cond::IOVNames::iovMappingVersion(),
                              cond::IOVNames::iovMappingXML());

    cond::MetaData  metadata( destdb );
    if( metadata.hasTag(destTag) ){
      cond::MetaDataEntry entry;
      metadata.getEntryByTag(destTag,entry);
      destiovtoken=entry.iovtoken;
      if (sourceiovtype!=entry.timetype) {
        throw std::runtime_error("iov type in source and dest differs");
      }
    }
    transaction.commit();
    if(debug){
      std::cout<<"destintion iov token "<< destiovtoken<<std::endl;
    }
  }

  bool newIOV = destiovtoken.empty();
  cond::IOVService iovmanager(sourcedb);

  sourcedb.transaction().start(true);
  std::string payloadContainer=iovmanager.payloadContainerName(sourceiovtoken);
  iovmanager.loadDicts(sourceiovtoken);
  sourcedb.transaction().commit();

  if (newIOV) {
    // store payload mapping

    sourcedb.transaction().start(true);
    {
      cond::DbScopedTransaction transaction(destdb);
      transaction.start(false);
      bool stored = destdb.importMapping( sourcedb, payloadContainer );
      if(debug)
        std::cout<< "payload mapping " << (stored ? "" : "not ") << "stored"<<std::endl;
      transaction.commit();
    }
    sourcedb.transaction().commit();
  }

  since = std::max(since, cond::timeTypeSpecs[sourceiovtype].beginValue);
  till  = std::min(till,  cond::timeTypeSpecs[sourceiovtype].endValue);

  int oldSize=0;
  if (!newIOV) {
    // grab info
    destdb.transaction().start(true);
    cond::IOVService iovmanager2(destdb);
    std::auto_ptr<cond::IOVIterator> iit(iovmanager2.newIOVIterator(destiovtoken,cond::IOVService::backwardIter));
    iit->next(); // just to initialize
    oldSize=iit->size();
    destdb.transaction().commit();
  }

  // setup logDB
  std::auto_ptr<cond::Logger> logdb;
  if (doLog) {
    cond::DbSession logSession = openDbSession( "logDB");
    logdb.reset(new cond::Logger(logSession));
    //   logdb->getWriteLock();
    logdb->createLogDBIfNonExist();
    // logdb->releaseWriteLock();
  }

  cond::UserLogInfo a;
  a.provenance=sourceConnect+"/"+inputTag;
  a.usertext="exportIOV V2.0;";
  {
    std::ostringstream ss;
    ss << "since="<< since <<", till="<< till << ", " << usertext << ";";
    a.usertext +=ss.str();
  }

  sourcedb.transaction().start(true);
  {
    cond::DbScopedTransaction transaction(destdb);
    transaction.start(false);
    destiovtoken=iovmanager.exportIOVRangeWithPayload( destdb,
                                                       sourceiovtoken,
                                                       destiovtoken,
                                                       since, till,
                                                       outOfOrder );
    transaction.commit();
  }
  sourcedb.transaction().commit();
  if (newIOV) {
    cond::MetaData destMetadata(destdb);
    cond::DbScopedTransaction transaction(destdb);
    transaction.start(false);
    destMetadata.addMapping(destTag,destiovtoken,sourceiovtype);
    if(debug){
      std::cout<<"dest iov token "<<destiovtoken<<std::endl;
      std::cout<<"dest iov type "<<sourceiovtype<<std::endl;
    }
    transaction.commit();
  }

  ::sleep(1);

  // grab info
  destdb.transaction().start(true);
  cond::IOVService iovmanager2(destdb);
  cond::IOVIterator* iit=iovmanager2.newIOVIterator(destiovtoken,cond::IOVService::backwardIter);
  std::string const & timetypestr = cond::timeTypeSpecs[sourceiovtype].name;
  iit->next(); // just to initialize
  cond::TagInfo result;
  result.name=destTag;
  result.token=destiovtoken;
  result.lastInterval=iit->validity();
  result.lastPayloadToken=iit->payloadToken();
  result.size=iit->size();
  delete iit;
  destdb.transaction().commit();

  {
    std::ostringstream ss;
    ss << "copied="<< result.size-oldSize <<";";
    a.usertext +=ss.str();
  }

  if (doLog){
    logdb->getWriteLock();
    logdb->logOperationNow(a,destConnect,result.lastPayloadToken,destTag,timetypestr,result.size-1);
    logdb->releaseWriteLock();
  }

  return 0;
}

int main( int argc, char** argv ){

  cond::ExportIOVUtilities utilities;
  return utilities.run(argc,argv);
}

