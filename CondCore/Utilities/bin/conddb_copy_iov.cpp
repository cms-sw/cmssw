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
  addOption<std::string>("description","x","user text (for new tags, optional)");
}

cond::CopyIovUtilities::~CopyIovUtilities(){
}

int cond::CopyIovUtilities::execute(){

  bool debug = hasDebug();
  std::string connect = getOptionValue<std::string>("connect");

  // this is mandatory
  std::string inputTag = getOptionValue<std::string>("inputTag");
  std::string tag = inputTag;
  if( hasOptionValue("tag")) tag = getOptionValue<std::string>("tag");

  cond::Time_t sourceSince = getOptionValue<cond::Time_t>( "sourceSince");
  cond::Time_t destSince = sourceSince;
  if( hasOptionValue("destSince") ) destSince = getOptionValue<cond::Time_t>( "destSince");

  std::string description("");
  if( hasOptionValue("description") ) description = getOptionValue<std::string>( "description" );

  persistency::ConnectionPool connPool;
  if( hasOptionValue("authPath") ){
    connPool.setAuthenticationPath( getOptionValue<std::string>( "authPath") ); 
  }
  connPool.configure();

  std::cout <<"# Connecting to source database on "<<connect<<std::endl;
  persistency::Session session = connPool.createSession( connect, true );

  std::cout <<"# input tag is "<<inputTag<<std::endl;
  std::cout <<"# destination tag is "<<tag<<std::endl;

  bool imported = copyIov( session, inputTag, tag, sourceSince, destSince, description );
    
  if( imported ) {
    std::cout <<"# 1 iov copied. "<<std::endl;
 }
  return 0;
}

int main( int argc, char** argv ){

  cond::CopyIovUtilities utilities;
  return utilities.run(argc,argv);
}

