#include "CondCore/DBCommon/interface/TokenBuilder.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "POOLCore/Token.h"
#include "StorageSvc/DbType.h"
#include "StorageSvc/DbReflex.h"
#include "SealBase/SharedLibrary.h"
#include "SealBase/SharedLibraryError.h"
#include "SealKernel/Exception.h"
//#include <iostream>
namespace cond{
  TokenBuilder::TokenBuilder(): m_token(new pool::Token){
    m_token->setTechnology(pool::POOL_RDBMS_StorageType.type());
  }
  TokenBuilder::~TokenBuilder(){
    delete m_token;
  }
  void TokenBuilder::set(const std::string& fid,
			 const std::string& dictLib,
			 const std::string& className,
			 const std::string& containerName,
			 int pkcolumnValue){
    try {  
      seal::SharedLibrary::load( "lib" + dictLib + ".so" );
    }catch ( seal::SharedLibraryError *error){
      throw cond::Exception(std::string("TokenBuilder::set failed loading dictionary ")+error->explainSelf());
    }catch (const seal::Exception &e){
      throw cond::Exception(std::string("TokenBuilder::set failed loading dictionary")+e.what());
    }catch (...){
      throw cond::Exception(std::string("TokenBuilder::set failed loading dictionary"));
    }
    ROOT::Reflex::Type myclass=ROOT::Reflex::Type::ByName(className);
    m_token->setDb(fid);
    m_token->setClassID(pool::DbReflex::guid(myclass));
    m_token->setCont(containerName);
    m_token->oid().first=0;
    m_token->oid().second=pkcolumnValue;
  }
  void TokenBuilder::resetOID( int pkcolumnValue ){
    m_token->oid().first=0;
    m_token->oid().second=pkcolumnValue;
  }
  std::string TokenBuilder::tokenAsString() const{
    return m_token->toString();
  }
}//ns cond
