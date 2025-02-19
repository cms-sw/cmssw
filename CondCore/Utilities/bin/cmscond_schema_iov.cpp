#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Auth.h"
#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/IOVService/interface/IOVSchemaUtility.h"
#include "CondCore/Utilities/interface/Utilities.h"
#include <iostream>

namespace cond {
  class SchemaIOVUtilities : public Utilities {
    public:
      SchemaIOVUtilities();
      ~SchemaIOVUtilities();
      int execute();
  };
}

cond::SchemaIOVUtilities::SchemaIOVUtilities():Utilities("cmscond_schema_iov"){
  addConnectOption();
  addAuthenticationOptions();
  addOption<bool>("create","","create iov schema");
  addOption<bool>("drop","","drop iov schema");
  //addOption<bool>("truncate","","truncate iov schema");
}

cond::SchemaIOVUtilities::~SchemaIOVUtilities(){
}

int cond::SchemaIOVUtilities::execute(){

  bool dropSchema= hasOptionValue("drop");
  bool createSchema= hasOptionValue("create");
  
  cond::DbSession session = openDbSession("connect", Auth::COND_ADMIN_ROLE );

  cond::IOVSchemaUtility util( session, std::cout );

  cond::DbScopedTransaction trans( session );
  if( dropSchema || createSchema ){
    trans.start(false);
    if( dropSchema ){
      util.dropIOVContainer();
    }
    if( createSchema ){
      util.createIOVContainer();
    }
    trans.commit();
    return 0;
  } 

  throw cond::Exception("Option create or drop not provided.");
}

int main( int argc, char** argv ){

  cond::SchemaIOVUtilities utilities;
  return utilities.run(argc,argv);
}

