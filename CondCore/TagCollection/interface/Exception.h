#ifndef CondCore_TagCollection_Exception_h
#define CondCore_TagCollection_Exception_h
#include "CondCore/DBCommon/interface/Exception.h"
#include <string>
namespace cond{
  class nonExistentGlobalTagInventoryException : public Exception{
  public:
    nonExistentGlobalTagInventoryException( const std::string& source ) : Exception( source+std::string(": global tag inventory does not exist") ){
    }
    virtual ~nonExistentGlobalTagInventoryException() throw(){
    }
  };

  class nonExistentGlobalTagException : public Exception{
  public:
    nonExistentGlobalTagException( const std::string& source , const std::string& globaltagName ) : Exception( source+std::string(": global tag: "+globaltagName+" does not exist") ){
    }
    virtual ~nonExistentGlobalTagException() throw(){
    }
  };
}
#endif
