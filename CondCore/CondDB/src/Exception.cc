#include "CondCore/CondDB/interface/Exception.h"

namespace cond {

  namespace persistency {


    Exception::Exception( const std::string& message ):
      cms::Exception( "ConditionDatabase", message ){}

    Exception::Exception( const std::string& message, const std::string& methodName ):
      cms::Exception( "ConditionDatabase", message+" from "+methodName ){}

    void throwException( const std::string& message, 
			 const std::string& methodName ){
      throw Exception( message, methodName );
    }

  }

  void throwException( const std::string& message, 
		       const std::string& methodName ){
    throw Exception( message, methodName );
  }
 
}
