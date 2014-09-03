#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "CondCore/CondDB/interface/Utils.h"
#include "CondCore/CondDB/interface/IOVEditor.h"
#include "CondCore/CondDB/interface/IOVProxy.h"

#include "CondCore/Utilities/interface/Utilities.h"
#include "CondCore/Utilities/interface/CondDBTools.h"
#include <iostream>

#include <sstream>

namespace cond {

  class CopyIovUtilities : public cond::Utilities {
    public:
      CopyIovUtilities();
      ~CopyIovUtilities();
      int execute();
  };
}

cond::CopyIovUtilities::CopyIovUtilities():Utilities("conddb_copy_iov"){
  addConnectOption("connect","c","target connection string (required)");
  addAuthenticationOptions();
  addOption<std::string>("inputTag","i","source tag (optional)");
  addOption<std::string>("tag","t","destination tag (optional)");
  addOption<cond::Time_t>("sourceSince","s","since time of the iov to copy (required)");
  addOption<cond::Time_t>("destSince","d","since time of the destination iov (optional, default=sourceSince)");
  addOption<std::string>("oraDestAccount","A","ora DB destination account (optional, to be used with -T -I)");
  addOption<std::string>("oraInputTag","I","ora DB input tag (optional, to be used with -A -T)");
  addOption<std::string>("oraDestTag","T","ora DB destination tag (optional, to be used with -A -I)");
}

cond::CopyIovUtilities::~CopyIovUtilities(){
}

int cond::CopyIovUtilities::execute(){

  bool debug = hasDebug();
  std::string connect = getOptionValue<std::string>("connect");

  std::string inputTag("");
  std::string tag("");
  std::string oraConn("");
  std::string oraTag("");
  std::string oraInputTag("");
  if( hasOptionValue("tag")) {
    tag = getOptionValue<std::string>("tag");
    inputTag = getOptionValue<std::string>("inputTag");
  } else {
    if( hasOptionValue("oraDestTag") ){
      oraTag = getOptionValue<std::string>("oraDestTag");
    } else throwException("The destination tag is missing and can't be resolved.","CopyIovUtilities::execute");
    oraInputTag = getOptionValue<std::string>("oraInputTag");
    oraConn = getOptionValue<std::string>("oraDestAccount");
  }

  cond::Time_t sourceSince = getOptionValue<cond::Time_t>( "sourceSince");
  cond::Time_t destSince = sourceSince;
  if( hasOptionValue("destSince") ) destSince = getOptionValue<cond::Time_t>( "destSince");

  persistency::ConnectionPool connPool;
  if( hasOptionValue("authPath") ){
    connPool.setAuthenticationPath( getOptionValue<std::string>( "authPath") ); 
  }
  connPool.configure();

  std::cout <<"# Connecting to source database on "<<connect<<std::endl;
  persistency::Session session = connPool.createSession( connect, true );

  bool newTag = false;
  session.transaction().start( true );
  if( tag.empty() ){
    cond::MigrationStatus ms = ERROR;
    std::cout <<"# checking TAG_MAPPING table to identify the input tag."<<std::endl;
    session.checkMigrationLog( oraConn, oraInputTag, inputTag, ms );
    if( ms == ERROR ) throwException("The input Tag is not mapped to any available tag.","CopyIovUtilities::execute");
    else if( ms == MIGRATED ) std::cout <<"# WARNING: the input tag has not been validated."<<std::endl;
    std::cout <<"# checking TAG_MAPPING table to identify the destination tag."<<std::endl;
    if( !session.checkMigrationLog( oraConn, oraTag, tag, ms ) ){
      tag = oraTag;
      newTag = true;
    } else {
      if( ms == ERROR ) throwException("The destination Tag has not been correctly migrated.","CopyIovUtilities::execute");
      else if( ms == MIGRATED ) std::cout <<"# WARNING: the destination tag has not been validated."<<std::endl;
    }
  }
  session.transaction().commit();

  std::cout <<"# input tag is "<<inputTag<<std::endl;
  std::cout <<"# destination tag is "<<tag<<std::endl;

  bool imported = copyIov( session, inputTag, tag, sourceSince, destSince, "", true );
    
  if( imported ) {
    std::cout <<"# 1 iov copied. "<<std::endl;
    if( newTag ){
      std::cout <<"# updating TAG_MAPPING table..."<<std::endl;
      session.transaction().start( false );
      session.addToMigrationLog( oraConn, oraTag, tag, VALIDATED );
      session.transaction().commit();      
    }
  }
  return 0;
}

int main( int argc, char** argv ){

  cond::CopyIovUtilities utilities;
  return utilities.run(argc,argv);
}

