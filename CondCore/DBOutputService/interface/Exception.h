#ifndef CondCore_DBOutputService_Exception_h
#define CondCore_DBOutputService_Exception_h
#include "CondCore/CondDB/interface/Exception.h"
#include <string>
namespace cond{
  class UnregisteredRecordException : public Exception{
  public:
    UnregisteredRecordException( const std::string& recordName ) : Exception( std::string("PoolDBOutputService: unregistered record "+recordName) ){
    }
    virtual ~UnregisteredRecordException() throw(){
    }
  };
}
#endif
