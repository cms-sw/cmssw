#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/DbScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/Auth.h"
#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/IOVService/interface/IOVSchemaUtility.h"
#include "CondCore/Utilities/interface/Utilities.h"
#include <iostream>

namespace cond {
  class SchemaManager : public Utilities {
    public:
      SchemaManager();
      ~SchemaManager();
      int execute();
  };
}

cond::SchemaManager::SchemaManager():Utilities("cmscond_schema_manager"){
  addConnectOption();
  addAuthenticationOptions();
  addOption<bool>("create","","create cond schema (iov by default)");
  addOption<bool>("drop","","drop cond schema");
  addOption<bool>("dropAll","","drop the entire schema");
  addOption<std::string>("payload","n","payload container name");
  addOption<std::string>("type","t","payload type");
}

cond::SchemaManager::~SchemaManager(){
}

int cond::SchemaManager::execute(){

  bool drop= hasOptionValue("drop");
  bool dropAll= hasOptionValue("dropAll");
  bool create= hasOptionValue("create");
  std::string payloadName("");
  if( hasOptionValue("payload") ) payloadName = getOptionValue<std::string>("payload");
  std::string typeName("");
  if( hasOptionValue("type") ) typeName = getOptionValue<std::string>("type");

  cond::DbSession session = openDbSession("connect", Auth::COND_ADMIN_ROLE);

  cond::IOVSchemaUtility util( session, std::cout );

  cond::DbScopedTransaction trans( session );

  if( dropAll ){
    trans.start(false);
    util.dropAll();
    trans.commit();
    return 0;
  }

  if( drop || create ){
    trans.start(false);
    if( drop ){
      if(payloadName.empty()){
	throw cond::Exception("Payload name not provided.");
      }
      util.dropPayloadContainer( payloadName );
    }
    if( create ){
      util.createIOVContainer();
      if(!payloadName.empty()) util.createPayloadContainer( payloadName, typeName );
    }
    trans.commit();
    return 0;
  }
  
  throw cond::Exception("Option create or drop not provided.");
}

int main( int argc, char** argv ){

  cond::SchemaManager utilities;
  return utilities.run(argc,argv);
}

