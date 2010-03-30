#ifndef INCLUDE_ORA_EXCEPTION_H
#define INCLUDE_ORA_EXCEPTION_H

#include "CondCore/DBCommon/interface/Exception.h"

namespace ora {

  /// Base exception class for the object to relational access 
  class Exception : public cond::Exception {
  public:
    /// Constructor
    Exception( const std::string& message, const std::string& methodName );
    /// Destructor
    virtual ~Exception() throw() {}
  };

  void throwException( const std::string& message, const std::string& methodName );

}

#endif
