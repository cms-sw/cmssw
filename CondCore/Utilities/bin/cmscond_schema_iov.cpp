#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
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
  addOption<bool>("truncate","","truncate iov schema");
}

cond::SchemaIOVUtilities::~SchemaIOVUtilities(){
}

int cond::SchemaIOVUtilities::execute(){

  bool dropSchema= hasOptionValue("drop");
  bool createSchema= hasOptionValue("create");
  bool truncateSchema= hasOptionValue("truncate");
  
  cond::DbSession session = openDbSession("connect");

  if( createSchema ){
    cond::DbScopedTransaction transaction(session);
    transaction.start(false);
    cond::IOVSchemaUtility ut(session);
    ut.create();
    transaction.commit();
    return 0;
  }
  if( dropSchema ){
    cond::DbScopedTransaction transaction(session);
    transaction.start(false);
    cond::IOVSchemaUtility ut(session);
    ut.drop();
    transaction.commit();
    return 0;
  }
  if( truncateSchema ){
    cond::DbScopedTransaction transaction(session);
    transaction.start(false);
    cond::IOVSchemaUtility ut(session);
    ut.truncate();
    transaction.commit();
    return 0;
  }
  return 0;
}

int main( int argc, char** argv ){

  cond::SchemaIOVUtilities utilities;
  return utilities.run(argc,argv);
}

