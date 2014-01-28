#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/MetaDataService/interface/MetaDataSchemaUtility.h"
#include "CondCore/Utilities/interface/Utilities.h"
#include <iostream>

namespace cond {
  class SchemaMetadataUtilities : public Utilities {
    public:
      SchemaMetadataUtilities();
      ~SchemaMetadataUtilities();
      int execute();
  };
}

cond::SchemaMetadataUtilities::SchemaMetadataUtilities():Utilities("cmscond_schema_metadata"){
  addConnectOption();
  addAuthenticationOptions();
  addOption<bool>("create","","create metadata schema");
  addOption<bool>("drop","","drop metadata schema");
}

cond::SchemaMetadataUtilities::~SchemaMetadataUtilities(){
}

int cond::SchemaMetadataUtilities::execute(){

  bool dropSchema= hasOptionValue("drop");
  bool createSchema= hasOptionValue("create");
  std::string tag;
  cond::DbSession session = openDbSession("connect");
  if( createSchema ){
    cond::DbScopedTransaction transaction(session);
    transaction.start(false);
    cond::MetaDataSchemaUtility ut( session );
    ut.create();
    transaction.commit();
    return 0;
  }
  if( dropSchema ){
    cond::DbScopedTransaction transaction(session);
    transaction.start(false);
    cond::MetaDataSchemaUtility ut( session );
    ut.drop();
    transaction.commit();
    return 0;
  }
  return 0;
}

int main( int argc, char** argv ){
  cond::SchemaMetadataUtilities utilities;
  return utilities.run(argc,argv);
}


