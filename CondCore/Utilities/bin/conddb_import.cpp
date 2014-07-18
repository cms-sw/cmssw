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
}

cond::ImportUtilities::~ImportUtilities(){
}

int cond::ImportUtilities::execute(){

  bool debug = hasDebug();
  std::string destConnect = getOptionValue<std::string>("connect" );
  std::string sourceConnect = getOptionValue<std::string>("fromConnect");

  std::string tag = getOptionValue<std::string>("tag");
  std::string inputTag = tag;
  if( hasOptionValue("inputTag")) {
    inputTag = getOptionValue<std::string>("inputTag");
  }

  cond::Time_t begin = 1;
  if( hasOptionValue("begin") ) begin = getOptionValue<cond::Time_t>( "begin");
  cond::Time_t end = cond::time::MAX_VAL;
  if( hasOptionValue("end") ) end = getOptionValue<cond::Time_t>( "end");
  if( begin > end ) throwException( "Begin time can't be greater than end time.","ImportUtilities::execute");


  persistency::ConnectionPool connPool;
  std::cout <<"# Connecting to source database on "<<sourceConnect<<std::endl;
  persistency::Session sourceSession = connPool.createSession( sourceConnect );

  std::cout <<"# Opening session on destination database..."<<std::endl;
  persistency::Session destSession = connPool.createSession( destConnect, true, COND_DB );
  
  size_t nimported = importIovs( inputTag, sourceSession, tag, destSession, begin, end, true );
    
  std::cout <<"# "<<nimported<<" iov(s) imported. "<<std::endl;

  return 0;
}

int main( int argc, char** argv ){

  cond::ImportUtilities utilities;
  return utilities.run(argc,argv);
}

