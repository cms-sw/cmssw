#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/ORA/interface/ScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/IOVService/interface/IOVNames.h"
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

  cond::DbSession session = openDbSession("connect");

  if( !drop && !create && !dropAll ){
    throw cond::Exception("Option create or drop not provided.");
  }

  if( create ){
    ora::Database& db = session.storage();
    ora::ScopedTransaction trans( db.transaction() );
    trans.start(false);
    if( !db.exists() ){
      std::cout << "INFO: Creating database "<<std::endl;
      db.create(cond::DbSession::COND_SCHEMA_VERSION);
      db.setAccessPermission(cond::DbSession::CONDITIONS_GENERAL_READER, false );
      db.setAccessPermission( cond::DbSession::CONDITIONS_GENERAL_WRITER, true );
    } 
    std::set<std::string> conts = db.containers();
    if( conts.find( cond::IOVNames::container() )==conts.end() ){
      std::cout << "INFO: Creating container \"" << cond::IOVNames::container() << "\"."<<std::endl;
      ora::Container c = db.createContainer( cond::IOVNames::container(), cond::IOVNames::container() );
      c.setAccessPermission( cond::DbSession::CONDITIONS_GENERAL_READER, false );
      c.setAccessPermission( cond::DbSession::CONDITIONS_GENERAL_WRITER, true );
    } else {
      std::cout << "WARNING: container \"" << cond::IOVNames::container() << "\" already exists in the database."<<std::endl;
    }

    if(!payloadName.empty()){
      if(typeName.empty()) throw cond::Exception("Typename for payload not provided.");
      std::set<std::string> conts = db.containers();
      if( conts.find( payloadName ) != conts.end()) throw cond::Exception("Container \""+payloadName+"\" already exists.");
      std::cout << "INFO: Creating container \"" << payloadName << "\"."<<std::endl;
      ora::Container c = db.createContainer( typeName, payloadName );
      c.setAccessPermission( cond::DbSession::CONDITIONS_GENERAL_READER, false );
      c.setAccessPermission( cond::DbSession::CONDITIONS_GENERAL_WRITER, true );
    }
          
    trans.commit();
    return 0;
  } 
  if( dropAll ){
    ora::Database& db = session.storage();
    ora::ScopedTransaction trans( db.transaction() );
    trans.start(false);
    if( !db.exists() ){
      std::cout << "ERROR: Condition database does not exist."<<std::endl;
      return 0;
    }
    std::cout << "INFO: Dropping database "<<std::endl;
    db.drop();
    trans.commit();
    return 0;
  }
  if( drop ){
    if(payloadName.empty()){
      throw cond::Exception("Payload name not provided.");
    }
    ora::Database& db = session.storage();
    ora::ScopedTransaction trans( db.transaction() );
    trans.start(false);
    if( !db.exists() ){
      std::cout << "ERROR: Condition database does not exist."<<std::endl;
      return 0;
    }
    std::set<std::string> conts = db.containers();
    if( conts.find( payloadName )==conts.end() ){
      std::cout << "WARNING: container \"" << payloadName << "\" does not exist in the database."<<std::endl;
      return 0;
    }
     std::cout << "INFO: Dropping container \"" << payloadName << "\"."<<std::endl;
   db.dropContainer( payloadName );
    trans.commit();
    return 0;
  }
  return 0;
}

int main( int argc, char** argv ){

  cond::SchemaManager utilities;
  return utilities.run(argc,argv);
}

