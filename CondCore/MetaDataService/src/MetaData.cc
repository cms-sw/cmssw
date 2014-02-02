#include "CondCore/MetaDataService/interface/MetaData.h"
#include "CondCore/ORA/interface/Database.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "RelationalAccess/SchemaException.h"
#include <memory>
//#include <iostream>


namespace {

    std::string mdErrorPrefix(const std::string& source, const std::string& name) {
      return source+std::string(": metadata entry \"")+name+std::string("\" ");
    }
    

    void mdError(const std::string& source, const std::string& name, const std::string& mess) {
      throw cond::Exception(mdErrorPrefix(source,name)+mess);
    }
    
    void mdDuplicateEntryError(const std::string& source, const std::string& name) {
      mdError(source, name, "Already exists");
    }

    void mdNoEntry(const std::string& source, const std::string& name) {
      mdError(source, name, "does not exists");
    }

}


cond::MetaData::MetaData(cond::DbSession& userSession):m_userSession( userSession ){
}
cond::MetaData::~MetaData(){
}
bool 
cond::MetaData::addMapping(const std::string& name, 
                           const std::string& iovtoken, 
                           cond::TimeType ){
  try{
    ora::OId oid;
    oid.fromString( iovtoken );
    m_userSession.storage().setObjectName( name, oid );
  }catch( const coral::DuplicateEntryInUniqueKeyException& er ){
    mdDuplicateEntryError("addMapping",name);
  }catch(std::exception& er){
    mdError("MetaData::addMapping",name,er.what());
  }
  return true;
}

const std::string 
cond::MetaData::getToken( const std::string& name ) const{
  bool ok=false;
  std::string iovtoken("");
  try{
    ora::OId oid;
    ok = m_userSession.storage().getItemId( name, oid );
    if(ok) {
      iovtoken = oid.toString();
    }
  }catch(const std::exception& er){
    mdError("MetaData::getToken", name,er.what() );
  }
  if (!ok) mdNoEntry("MetaData::getToken", name);
  return iovtoken;
}

bool cond::MetaData::hasTag( const std::string& name ) const{
  bool result=false;
  try{
    ora::OId oid;
    result = m_userSession.storage().getItemId( name, oid );
  }catch(const std::exception& er){
    mdError("MetaData::hasTag", name, er.what() );
  }
  return result;
}

void 
cond::MetaData::listAllTags( std::vector<std::string>& result ) const{
  try{
    m_userSession.storage().listObjectNames( result );
  }catch(const std::exception& er){
    throw cond::Exception( std::string("MetaData::listAllTags: " )+er.what() );
  }
}

void 
cond::MetaData::deleteAllEntries(){
  try{
    m_userSession.storage().eraseAllNames();
  }catch(const std::exception& er){
    throw cond::Exception( std::string("MetaData::deleteAllEntries: " )+er.what() );
  }
}

void cond::MetaData::deleteEntryByTag( const std::string& tag ){
  try{
    m_userSession.storage().eraseObjectName( tag );   
  }catch(const std::exception& er){
    throw cond::Exception( std::string("MetaData::deleteEntryByTag: " )+er.what() );
  }
}
