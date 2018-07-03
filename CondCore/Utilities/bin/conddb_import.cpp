#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/CondDB/interface/Utils.h"
#include "CondCore/CondDB/interface/IOVEditor.h"
#include "CondCore/CondDB/interface/IOVProxy.h"

#include "CondCore/Utilities/interface/Utilities.h"
#include "CondCore/Utilities/interface/CondDBTools.h"
#include <iostream>

#include <sstream>

namespace cond {

  class ImportUtilities : public cond::Utilities {
    public:
      ImportUtilities();
      ~ImportUtilities();
      int execute();
  };
}

cond::ImportUtilities::ImportUtilities():Utilities("conddb_import"){
  addConnectOption("fromConnect","f","source connection string (optional, default=connect)");
  addConnectOption("connect","c","target connection string (required)");
  addAuthenticationOptions();
  addOption<std::string>("inputTag","i","source tag (optional - default=tag)");
  addOption<std::string>("tag","t","destination tag (required)");
  addOption<cond::Time_t>("begin","b","lower bound of interval to import (optional, default=1)");
  addOption<cond::Time_t>("end","e","upper bound of interval to import (optional, default=infinity)");
  addOption<std::string>("description","x","User text ( for new tags, optional )");
  addOption<bool>("override","o","Override the existing iovs in the dest tag, for the selected interval ( optional, default=false)");
  addOption<bool>("reserialize","r","De-serialize in reading and serialize in writing (optional, default=false)");
  addOption<bool>("forceInsert","K","force the insert for all synchronization types (optional, default=false)");
  addOption<std::string>("editingNote","N","editing note (required with forceInsert)");
}

cond::ImportUtilities::~ImportUtilities(){
}

int cond::ImportUtilities::execute(){

  bool debug = hasDebug();
  std::string destConnect = getOptionValue<std::string>("connect" );
  std::string sourceConnect = destConnect;
  if(hasOptionValue("fromConnect")) sourceConnect = getOptionValue<std::string>("fromConnect");

  std::string inputTag = getOptionValue<std::string>("inputTag");;
  std::string tag = inputTag;
  if( hasOptionValue("tag")) {
    tag = getOptionValue<std::string>("tag");
  }

  std::string description("");
  if( hasOptionValue("description") ) description = getOptionValue<std::string>("description");
  bool override = hasOptionValue("override");
  bool reserialize = hasOptionValue("reserialize");
  bool forceInsert = hasOptionValue("forceInsert");
  std::string editingNote("");
  if( hasOptionValue("editingNote") ) editingNote = getOptionValue<std::string>("editingNote");
  if( forceInsert && editingNote.empty() ) {
    std::cout <<"ERROR: \'forceInsert\' mode requires an \'editingNote\' to be provided."<<std::endl;
    return -1;
  }

  cond::Time_t begin = 1;
  if( hasOptionValue("begin") ) begin = getOptionValue<cond::Time_t>( "begin");
  cond::Time_t end = cond::time::MAX_VAL;
  if( hasOptionValue("end") ) end = getOptionValue<cond::Time_t>( "end");
  if( begin > end ) throwException( "Begin time can't be greater than end time.","ImportUtilities::execute");

  persistency::ConnectionPool connPool;
  if( hasOptionValue("authPath") ){
    connPool.setAuthenticationPath( getOptionValue<std::string>( "authPath") ); 
  }
  connPool.configure();
  std::cout <<"# Running import tool for conditions on release "<<cond::currentCMSSWVersion()<<std::endl;
  std::cout <<"# Connecting to source database on "<<sourceConnect<<std::endl;
  persistency::Session sourceSession = connPool.createSession( sourceConnect );

  std::cout <<"# Opening session on destination database..."<<std::endl;
  persistency::Session destSession = connPool.createSession( destConnect, true );

  std::cout <<"# destination tag is "<<tag<<std::endl;

  size_t nimported = importIovs( inputTag, sourceSession, tag, destSession, begin, end, description, editingNote, override, reserialize, forceInsert );
  std::cout <<"# "<<nimported<<" iov(s) imported. "<<std::endl;

  return 0;
}

int main( int argc, char** argv ){

  cond::ImportUtilities utilities;
  return utilities.run(argc,argv);
}

