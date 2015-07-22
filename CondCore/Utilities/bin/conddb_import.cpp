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
  addConnectOption("fromConnect","f","source connection string (required)");
  addConnectOption("connect","c","target connection string (required)");
  addAuthenticationOptions();
  addOption<std::string>("inputTag","i","source tag (optional - default=tag)");
  addOption<std::string>("tag","t","destination tag (required)");
  addOption<cond::Time_t>("begin","b","lower bound of interval to import (optional, default=1)");
  addOption<cond::Time_t>("end","e","upper bound of interval to import (optional, default=infinity)");
  addOption<std::string>("oraDestAccount","A","ora DB destination account (optional, to be used with -T)");
  addOption<std::string>("oraDestTag","T","ora DB destination tag (optional, to be used with -A)");
}

cond::ImportUtilities::~ImportUtilities(){
}

int cond::ImportUtilities::execute(){

  bool debug = hasDebug();
  std::string destConnect = getOptionValue<std::string>("connect" );
  std::string sourceConnect = getOptionValue<std::string>("fromConnect");

  std::string inputTag = getOptionValue<std::string>("inputTag");;
  std::string tag(""); 
  std::string oraConn("");
  std::string oraTag("");
  if( hasOptionValue("tag")) {
    tag = getOptionValue<std::string>("tag");
  } else {
    if( hasOptionValue("oraDestTag") ){
      oraTag = getOptionValue<std::string>("oraDestTag");
    } else throwException("The destination tag is missing and can't be resolved.","ImportUtilities::execute");; 
    oraConn = getOptionValue<std::string>("oraDestAccount");
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

  bool newTag = false;
  destSession.transaction().start( true );
  if( tag.empty() ){
    cond::MigrationStatus ms = ERROR;
    std::cout <<"# checking TAG_MAPPING table to identify destination tag."<<std::endl; 
    if( !destSession.checkMigrationLog( oraConn, oraTag, tag, ms ) ){
      tag = oraTag;
      newTag = true;
    } else {
      if( ms == ERROR ) throwException("The destination Tag has not been correctly migrated.","ImportUtilities::execute");
      else if( ms == MIGRATED ) std::cout <<"# WARNING: the destination tag has not been validated."<<std::endl;
    }
  }
  destSession.transaction().commit();

  std::cout <<"# destination tag is "<<tag<<std::endl;

  size_t nimported = importIovs( inputTag, sourceSession, tag, destSession, begin, end, "", true );
  std::cout <<"# "<<nimported<<" iov(s) imported. "<<std::endl;
  if( newTag && nimported ){
    std::cout <<"# updating TAG_MAPPING table..."<<std::endl;
    destSession.transaction().start( false );
    destSession.addToMigrationLog( oraConn, oraTag, tag, VALIDATED );
    destSession.transaction().commit();
  }

  return 0;
}

int main( int argc, char** argv ){

  cond::ImportUtilities utilities;
  return utilities.run(argc,argv);
}

