#ifndef INCLUDE_ORA_EXCEPTION_H
#define INCLUDE_ORA_EXCEPTION_H

#include "FWCore/Utilities/interface/Exception.h"

namespace ora {

  /// Base exception class for the object to relational access 
  class Exception : public cms::Exception {
  public:
    /// Constructor
    Exception( const std::string& message, const std::string& methodName );
    /// Destructor
    virtual ~Exception() throw() {}
  };

  void throwException( const std::string& message, const std::string& methodName );

  void throwException( const std::string& message, const std::type_info& sourceType, const std::string& methodName  );

}

#endif
