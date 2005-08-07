#include "CondCore/DBCommon/interface/TokenBuilder.h"
#include "POOLCore/Token.h"
#include "StorageSvc/DbType.h"
#include "SealBase/SharedLibrary.h"
#include "SealBase/SharedLibraryError.h"
#include "Reflection/Class.h"
#include "Reflection/PropertyList.h"
#include "SealKernel/Exception.h"
#include <stdexcept>
#include <iostream>
namespace cond{
  TokenBuilder::TokenBuilder(): m_token(new pool::Token){
    m_token->setTechnology(pool::POOL_RDBMS_StorageType.type());
  }
  TokenBuilder::~TokenBuilder(){
    delete m_token;
  }
  void TokenBuilder::setDB(const std::string& fid){
    m_token->setDb(fid);
  }
  void TokenBuilder::setContainer(const std::string& classguid,
				  const std::string& containerName){
    pool::Guid classid(classguid);
    m_token->setClassID(classid);
    m_token->setCont(containerName);
  }
  void TokenBuilder::setContainerFromDict(const std::string& dictLib,
				  const std::string& className,
				  const std::string& containerName){
    //m_token->setCont(containerName);
    std::string classid("");
    std::cout << "Loading library " << dictLib << std::endl;
    try {  // This is temporary. SEAL should get better in this.
      seal::SharedLibrary::load( "lib" + dictLib + ".so" );
    }catch ( seal::SharedLibraryError *error){
      std::cout << "test ERROR: "<< error->explainSelf() << std::endl;
    }catch (const seal::Exception &e){
      std::cout << "test ERROR: "<< e.what() << std::endl;
    }catch (const std::exception &e) {
      std::cout << "test ERROR: "<< e.what() << std::endl;
    }
    std::cout<<dictLib<<" loaded"<<std::endl;
    const seal::reflect::Class* classDictionary=seal::reflect::Class::forName(className);
    std::cout<<"Class *"<<classDictionary<<std::endl;
    //std::string classfullName = classDictionary->fullName();
    seal::reflect::PropertyList* properties=classDictionary->propertyList();
    if(properties){
      classid=properties->getProperty("ClassID");
      if(classid.empty()){
	std::cerr<<"Error: no id property found for class "<<className<<std::endl;
	throw std::runtime_error( "no id property found" );
      }
      std::cout<<"classid found "<<classid<<std::endl;
    }else{
      std::cerr<<"Error: no id property found for class "<<className<<std::endl;
      throw std::runtime_error( "no id property found" );
    }
    this->setContainer(classid, containerName);
  }
  void TokenBuilder::setOID(int pkcolumnValue ){
    m_token->oid().first=0;
    m_token->oid().second=pkcolumnValue;
  }
  std::string TokenBuilder::tokenAsString() const{
    return m_token->toString();
  }
}//ns cond
