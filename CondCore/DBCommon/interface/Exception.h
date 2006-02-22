#ifndef COND_EXCEPTION_H
#define COND_EXCEPTION_H
#include "FWCore/Utilities/interface/Exception.h"
#include <string>
namespace cond{
  class Exception : public cms::Exception{
  public:
    explicit Exception( const std::string& message );    
    virtual ~Exception();
  };
  class noDataForRequiredTimeException : public Exception{
  public:
    noDataForRequiredTimeException(const std::string& from,
				   const std::string& rcd,
				   const std::string& current);
    virtual ~noDataForRequiredTimeException();
  };
}
#endif
