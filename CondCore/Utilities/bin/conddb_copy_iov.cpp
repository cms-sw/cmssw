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
  addOption<std::string>("inputTag","i","source tag (required)");
  addOption<std::string>("tag","t","destination tag (required)");
  addOption<cond::Time_t>("sourceSince","s","since time of the iov to copy (required)");
  addOption<cond::Time_t>("destSince","d","since time of the destination iov (optional, default=sourceSince)");
}

cond::CopyIovUtilities::~CopyIovUtilities(){
}

int cond::CopyIovUtilities::execute(){

  bool debug = hasDebug();
  std::string connect = getOptionValue<std::string>("connect");

  std::string tag = getOptionValue<std::string>("tag");
  std::string inputTag = getOptionValue<std::string>("inputTag");

  cond::Time_t sourceSince = getOptionValue<cond::Time_t>( "sourceSince");
  cond::Time_t destSince = sourceSince;
  if( hasOptionValue("destSince") ) destSince = getOptionValue<cond::Time_t>( "destSince");

  persistency::ConnectionPool connPool;
  std::cout <<"# Connecting to source database on "<<connect<<std::endl;
  persistency::Session session = connPool.createSession( connect, true, COND_DB );

  bool imported = copyIov( session, inputTag, tag, sourceSince, destSince, true );
    
  if( imported ) std::cout <<"# 1 iov imported. "<<std::endl;

  return 0;
}

int main( int argc, char** argv ){

  cond::CopyIovUtilities utilities;
  return utilities.run(argc,argv);
}

