#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaData.h"

#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/IOVService/interface/IOVProxy.h"


#include "CondCore/DBCommon/interface/Logger.h"
#include "CondCore/DBCommon/interface/LogDBEntry.h"
#include "CondCore/DBCommon/interface/UserLogInfo.h"
#include "CondCore/DBCommon/interface/TagInfo.h"

#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/DBCommon/interface/IOVInfo.h"

#include "CondCore/Utilities/interface/Utilities.h"
#include <iterator>
#include <limits>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cstdlib>
#include<sstream>

namespace cond {
  class DuplicateIOVUtilities : public Utilities {
    public:
      DuplicateIOVUtilities();
      ~DuplicateIOVUtilities();
      int execute();
  };
}

cond::DuplicateIOVUtilities::DuplicateIOVUtilities():Utilities("cmscond_duplicate_iov"){
  addConnectOption();
  addAuthenticationOptions();
  addLogDBOption();
  addOption<std::string>("tag","t","tag (required)");
  addOption<std::string>("destTag","d","destination tag (if different than source tag)");
  addOption<cond::Time_t>("fromTime","f","a valid time of payload to append (required)");
  addOption<cond::Time_t>("sinceTime","s","since time of new iov(required)");
}

cond::DuplicateIOVUtilities::~DuplicateIOVUtilities(){
}

int cond::DuplicateIOVUtilities::execute(){

  std::string sourceTag = getOptionValue<std::string>("tag");
  std::string destTag(sourceTag);
  if(hasOptionValue("destTag")) destTag = getOptionValue<std::string>("destTag");
  cond::Time_t from = getOptionValue<cond::Time_t>("fromTime");
  cond::Time_t since = getOptionValue<cond::Time_t>("sinceTime");
  bool doLog = hasOptionValue("logDB");
  bool debug = hasDebug();
  std::string destConnect = getOptionValue<std::string>("connect" );
  
  cond::DbSession destDb = openDbSession( "connect" );

  std::string iovtoken("");
  std::string destiovtoken("");
  cond::TimeType iovtype;
  std::string timetypestr("");
   
  if(debug){
    std::cout << "source tag " << sourceTag << std::endl;
    std::cout << "dest   tag  " << destTag << std::endl;
  }  
    
  // find tag
  {
    destDb.transaction().start(true);
    cond::MetaData  metadata(destDb);
    {
      iovtoken = metadata.getToken(sourceTag);
      if(iovtoken.empty()) 
	throw std::runtime_error(std::string("tag ")+sourceTag+std::string(" not found") );
      cond::IOVService iovmanager(destDb);
      iovtype=iovmanager.timeType(iovtoken);
      timetypestr = cond::timeTypeSpecs[iovtype].name;
    }
    if( metadata.hasTag(destTag) ){
      destiovtoken=metadata.getToken(destTag);
      cond::IOVService iovmanager( destDb );
      if (iovtype!=iovmanager.timeType(destiovtoken)) {
        throw std::runtime_error("iov type in source and dest differs");
      }
    }
    destDb.transaction().commit();
    if(debug){
      std::cout<<"source iov token "<< iovtoken<<std::endl;
      std::cout<<"dest   iov token "<< destiovtoken<<std::endl;
      std::cout<<"iov type "<<  timetypestr<<std::endl;
    }
  }

  destDb.transaction().start(true);
  cond::IOVService iovmanager( destDb );
  std::string payload = iovmanager.payloadToken(iovtoken,from);
  destDb.transaction().commit();
  if (payload.empty()) {
    std::cerr <<"[Error] no payload found for time " << from << std::endl;
    return 1;
  };
  
  int size=0;
  bool newIOV = destiovtoken.empty();
  /* we allow multiple iov pointing to the same payload...
  if (!newIOV) {
    // to be streamlined
    cond::IOVProxy iov( destDb,destiovtoken,false,true);
    size = iov.size();
    if ( (iov.end()-1)->wrapperToken()==payload) {
      std::cerr <<"[Warning] payload for time " << from
                <<" equal to last inserted payload, no new IOV will be created" <<  std::endl;
      return 0;
    }
    if (payload == iovmanager.payloadToken(destiovtoken,since)) {
      std::cerr <<"[Warning] payload for time " << from
                <<" equal to payload valid at time "<< since
                <<", no new IOV will be created" <<  std::endl;
      return 0;
    }
  }
  */

  std::auto_ptr<cond::Logger> logdb;
  if(doLog){
    cond::DbSession logSession = openDbSession( "logDB" );
    // setup logDB
    logdb.reset(new cond::Logger( logSession ));
    //logdb->getWriteLock();
    logdb->createLogDBIfNonExist();
    //logdb->releaseWriteLock();
  }
  
  cond::UserLogInfo a;
  a.provenance=destConnect+"/"+destTag;
  a.usertext="duplicateIOV V1.0;";
  {
    std::ostringstream ss;
    ss << "from="<< from <<", since="<< since <<";";
    a.usertext +=ss.str();
  }

  // create if does not exist;
  if (newIOV) {
    cond::IOVEditor editor(destDb);
    cond::DbScopedTransaction transaction(destDb);
    transaction.start(false);
    editor.create(iovtype);
    destiovtoken=editor.token();
    editor.append(since,payload);
    transaction.commit();
  } else {
    //append it
    cond::IOVEditor editor(destDb,destiovtoken);
    cond::DbScopedTransaction transaction(destDb);
    transaction.start(false);
    editor.append(since,payload);
    editor.stamp(cond::userInfo(),false);
    transaction.commit();
  }

  if (newIOV) {
    cond::DbScopedTransaction transaction(destDb);
    cond::MetaData destMetadata( destDb );
    transaction.start(false);
    destMetadata.addMapping(destTag,destiovtoken,iovtype);
    if(debug){
      std::cout<<"dest iov token "<<destiovtoken<<std::endl;
      std::cout<<"dest iov type "<<iovtype<<std::endl;
    }
    transaction.commit();
  }
  
  ::sleep(1);
  
  if (doLog){
    logdb->getWriteLock();
    logdb->logOperationNow(a,destConnect,payload,destTag,timetypestr,size,since);
    logdb->releaseWriteLock();
  }

  return 0;
}

int main( int argc, char** argv ){

  cond::DuplicateIOVUtilities utilities;
  return utilities.run(argc,argv);
}

