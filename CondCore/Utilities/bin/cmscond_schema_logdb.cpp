#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Logger.h"
#include "CondCore/DBCommon/interface/Auth.h"
#include "CondCore/Utilities/interface/Utilities.h"
#include <iostream>

namespace cond {
  class SchemaLogDbUtilities : public Utilities {
    public:
      SchemaLogDbUtilities();
      ~SchemaLogDbUtilities();
      int execute();
  };
}

cond::SchemaLogDbUtilities::SchemaLogDbUtilities():Utilities("cmscond_schema_logdb"){
  addConnectOption();
  addAuthenticationOptions();
  addOption<bool>("create","","create logdb schema");
}

cond::SchemaLogDbUtilities::~SchemaLogDbUtilities(){
}

int cond::SchemaLogDbUtilities::execute(){

  bool createSchema= hasOptionValue("create");
  
  cond::DbSession session = openDbSession("connect", Auth::COND_ADMIN_ROLE );

  cond::Logger logger(session);

  cond::DbScopedTransaction trans( session );
  if( createSchema ){
    trans.start(false);
    logger.createLogDBIfNonExist();
    trans.commit();
    return 0;
  } 

  throw cond::Exception("Option create not provided.");
}

int main( int argc, char** argv ){

  cond::SchemaLogDbUtilities utilities;
  return utilities.run(argc,argv);
}


