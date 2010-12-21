#ifdef COND_EXP_WARNING
#warning please ignore previous warning
#endif
#ifndef COND_EXCEPTION_H
#define COND_EXCEPTION_H
#include "FWCore/Utilities/interface/Exception.h"
#include <string>
namespace cond{
  class Exception : public cms::Exception{
  public:
    explicit Exception( const std::string& message );    
    virtual ~Exception() throw();
  };
  class noDataForRequiredTimeException : public Exception{
  public:
    noDataForRequiredTimeException(const std::string& from,
				   const std::string& rcd,
				   const std::string& current);
    virtual ~noDataForRequiredTimeException() throw();
  };
  class RefException : public Exception{
  public:
    RefException(const std::string& from, 
		 const std::string& msg);
    virtual ~RefException() throw(){}
  };
  class TransactionException : public Exception{
  public:
    TransactionException(const std::string& from, 
			 const std::string& msg);
    virtual ~TransactionException() throw(){}
  };
  
  void throwException( std::string const& message, std::string const& methodName );
}
#endif
