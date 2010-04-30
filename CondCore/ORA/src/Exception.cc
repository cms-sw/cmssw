#include "CondCore/ORA/interface/Exception.h"

ora::Exception::Exception( const std::string& message, const std::string& methodName ):
  cms::Exception( message+" from "+methodName )
{}

namespace ora {
  void throwException( const std::string& message, const std::string& methodName ){
    throw Exception( message, methodName );
  }
  
}

