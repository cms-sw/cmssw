#include "CondCore/ORA/interface/Exception.h"
#include "ClassUtils.h"

ora::Exception::Exception( const std::string& message,
                           const std::string& methodName ):
  cms::Exception( message+" from "+methodName )
{}

namespace ora {
  void throwException( const std::string& message,
                       const std::string& methodName ){
    throw Exception( message, methodName );
  }

  void throwException( const std::string& message,
                       const std::type_info& sourceType,
                       const std::string& methodName ){
    std::string className = ClassUtils::demangledName( sourceType );
    throw Exception( message, className+"::"+methodName );
  }
  
}

