#include "CondCore/DBCommon/interface/TokenBuilder.h"
#include "POOLCore/Token.h"
#include "StorageSvc/DbType.h"
#include "StorageSvc/DbReflex.h"
#include "FWCore/PluginManager/interface/SharedLibrary.h"
//#include <iostream>

#if ROOT_VERSION_CODE < ROOT_VERSION(5,19,0)
 using namespace ROOT;
#endif

namespace cond{
  TokenBuilder::TokenBuilder(): m_token(new pool::Token){
    m_token->setTechnology(pool::POOL_RDBMS_HOMOGENEOUS_StorageType.type());
  }
  TokenBuilder::~TokenBuilder(){
    delete m_token;
  }
  void TokenBuilder::set(const std::string& fid,
			 const std::string& dictLib,
			 const std::string& className,
			 const std::string& containerName,
			 int pkcolumnValue){

    const boost::filesystem::path dict_path(dictLib);
    edmplugin::SharedLibrary shared( dict_path );
    Reflex::Type myclass=Reflex::Type::ByName(className);
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
