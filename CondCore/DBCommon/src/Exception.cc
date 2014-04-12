#include "CondCore/DBCommon/interface/Exception.h"
cond::Exception::Exception( const std::string& message ):cms::Exception("Conditions",message){}
cond::Exception::~Exception() throw(){}

cond::noDataForRequiredTimeException::noDataForRequiredTimeException(
const std::string& from, const std::string& rcd, const std::string& current ):
  cond::Exception(from+":  no data available in "+rcd+" at time "+current){}
cond::noDataForRequiredTimeException::~noDataForRequiredTimeException() throw(){}

cond::RefException::RefException( const std::string& from, const std::string& msg):
  cond::Exception(std::string("Error in building cond::Ref ")+from+" "+msg){}

cond::TransactionException::TransactionException( const std::string& from, const std::string& msg):
  cond::Exception(std::string("Transaction Error ")+from+" "+msg){}

namespace cond {
  void throwException( std::string const& message,
                       std::string const& methodName ){
    throw Exception( methodName + ": " + message );
  }
}
