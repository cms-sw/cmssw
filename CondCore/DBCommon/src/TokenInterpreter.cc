#include "CondCore/DBCommon/interface/TokenInterpreter.h"
#include "POOLCore/Token.h"
#include "StorageSvc/DbReflex.h"
namespace cond{
  TokenInterpreter::TokenInterpreter(const std::string& tokenString): m_tokenstr(tokenString){
    pool::Token* mytoken=new pool::Token;
    m_containerName=mytoken->fromString(tokenString).contID();
    const pool::Guid& classID=mytoken->fromString(tokenString).classID();
    ROOT::Reflex::Type myclass=pool::DbReflex::forGuid(classID);
    m_className=myclass.Name(); 
    //for the moment default name. Should decide FINAL,SCOPED or QUALIFIED
    mytoken->release();
  }
  TokenInterpreter::~TokenInterpreter(){
  }
  std::string TokenInterpreter::containerName()const{
    return m_containerName;
  }
  std::string TokenInterpreter::className()const{
    return m_className;
  }
}//ns cond
