#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/ORA/interface/ScopedTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/IOVService/interface/IOVNames.h"
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
  
  cond::DbSession session = openDbSession("connect");

  if( !dropSchema && !createSchema ){
    throw cond::Exception("Option create or drop not provided.");
  }

  if( createSchema ){
    ora::Database& db = session.storage();
    ora::ScopedTransaction trans( db.transaction() );
    trans.start(false);
    if( !db.exists() ){
      db.create(cond::DbSession::COND_SCHEMA_VERSION);
    } 
    std::set<std::string> conts = db.containers();
    if( conts.find( cond::IOVNames::container() )!=conts.end() ){
      std::cout << "WARNING: container \"" << cond::IOVNames::container() << "\" already exists in the database."<<std::endl;
      return 0;
    }
          
    db.createContainer( cond::IOVNames::container(), cond::IOVNames::container() );
    trans.commit();
    return 0;
  } 
  if( dropSchema ){
    ora::Database& db = session.storage();
    ora::ScopedTransaction trans( db.transaction() );
    trans.start(false);
    if( !db.exists() ){
      std::cout << "ERROR: Condition database does not exist."<<std::endl;
      return 0;
    }
    std::set<std::string> conts = db.containers();
    if( conts.find( cond::IOVNames::container() )==conts.end() ){
      std::cout << "WARNING: container \"" << cond::IOVNames::container() << "\" does not exist in the database."<<std::endl;
      return 0;
    }
    db.dropContainer( cond::IOVNames::container() );
    trans.commit();
    return 0;
  }
  return 0;
}

int main( int argc, char** argv ){

  cond::SchemaIOVUtilities utilities;
  return utilities.run(argc,argv);
}

