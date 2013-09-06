#include "CondCore/CondDB/interface/Exception.h"

conddb::Exception::Exception( const std::string& message, const std::string& methodName ):
  cms::Exception( "ConditionDatabase", message+" from "+methodName ){}

void conddb::throwException( const std::string& message, 
					const std::string& methodName ){
  throw Exception( message, methodName );
}
