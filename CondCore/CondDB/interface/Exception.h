#ifndef CondCore_CondDB_Exception_h
#define CondCore_CondDB_Exception_h

#include "FWCore/Utilities/interface/Exception.h"
#include <typeinfo>

namespace conddb {

  /// Base exception class for the object to relational access 
  class Exception : public cms::Exception {
  public:
    /// Constructor
    Exception( const std::string& message, const std::string& methodName );
    /// Destructor
    virtual ~Exception() throw() {}
  };

  void throwException( const std::string& message, const std::string& methodName );

}

#endif
