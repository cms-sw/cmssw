#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Auth.h"
#include "CondCore/MetaDataService/interface/MetaData.h"

#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/IOVService/interface/IOVProxy.h"


#include "CondCore/DBCommon/interface/Logger.h"
#include "CondCore/DBCommon/interface/LogDBEntry.h"
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
  addOption<std::string>("usertext","x","user text, to be included in usertext column (optional, must be enclosed in double quotes)");
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

  std::string usertext("no user comments");
  if( hasOptionValue("usertext")) usertext = getOptionValue<std::string>("usertext");
  
  cond::DbSession destDb = openDbSession( "connect", Auth::COND_WRITER_ROLE );

  std::string iovToken("");
  std::string destIovToken("");
  cond::TimeType iovType;
  std::string timetypestr("");
   
  if(debug){
    std::cout << "source tag " << sourceTag << std::endl;
    std::cout << "dest   tag  " << destTag << std::endl;
  }  
    
  cond::IOVProxy sourceIov(destDb);
  cond::IOVEditor destIov(destDb);
  // find tag
  cond::DbScopedTransaction transaction(destDb);
  transaction.start(false);
  destDb.storage().lockContainer(  IOVNames::container() );
  cond::MetaData  metadata(destDb);
  iovToken = metadata.getToken(sourceTag);
  if(iovToken.empty()) 
    throw std::runtime_error(std::string("tag ")+sourceTag+std::string(" not found") );
  sourceIov.load( iovToken );
  iovType=sourceIov.timetype();
  timetypestr = cond::timeTypeSpecs[iovType].name;
  if( metadata.hasTag(destTag) ){
    destIovToken=metadata.getToken(destTag);
    destIov.load( destIovToken );
    if (iovType!=destIov.proxy().timetype()) {
      throw std::runtime_error("iov type in source and dest differs");
    }
  } else {
    destIovToken = destIov.create( iovType );
    metadata.addMapping(destTag,destIovToken,iovType);
  }
  if(debug){
    std::cout<<"source iov token "<< iovToken<<std::endl;
    std::cout<<"dest   iov token "<< destIovToken<<std::endl;
    std::cout<<"iov type "<<  timetypestr<<std::endl;
  }
  cond::IOVProxy::const_iterator iElem = sourceIov.find( from );
  if( iElem == sourceIov.end() ){
    std::cerr <<"[Error] no payload found for time " << from << std::endl;
    return 1;    
  }
  std::string payload = iElem->token();
  std::string payloadClass = destDb.classNameForItem( payload );

  destIov.append(since,payload);
  destIov.stamp(cond::userInfo(),false);

  transaction.commit();

  ::sleep(1);
  
  // setup logDB and write on it...
  if (doLog){
    std::auto_ptr<cond::Logger> logdb;
    cond::DbSession logSession = openDbSession( "logDB", Auth::COND_WRITER_ROLE );
    logdb.reset(new cond::Logger( logSession ));
    logdb->createLogDBIfNonExist();

    cond::UserLogInfo a;
    a.provenance=destConnect+"/"+destTag;
    a.usertext="duplicateIOV V1.0;";
    std::ostringstream ss;
    ss << "From="<< from <<"; Since="<< since <<"; " << usertext;
    a.usertext +=ss.str();

    logdb->logOperationNow(a,destConnect,payloadClass,payload,destTag,timetypestr,0,since);
  }

  return 0;
}

int main( int argc, char** argv ){

  cond::DuplicateIOVUtilities utilities;
  return utilities.run(argc,argv);
}

