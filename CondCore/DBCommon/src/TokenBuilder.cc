#include "CondCore/DBCommon/interface/TokenBuilder.h"
#include "POOLCore/Token.h"
#include "StorageSvc/DbType.h"
#include "StorageSvc/DbReflex.h"
#include "SealBase/SharedLibrary.h"
#include "SealBase/SharedLibraryError.h"
#include "SealKernel/Exception.h"
#include <iostream>
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
      std::cerr << "caught seal::SharedLibraryError: "<< error->explainSelf() << std::endl;
    }catch (const seal::Exception &e){
      std::cout << "caught seal::Exception: "<< e.what() << std::endl;
    }
    seal::reflex::Type myclass=seal::reflex::Type::byName(className);
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
