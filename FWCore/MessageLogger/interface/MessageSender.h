#ifndef FWCore_MessageLogger_MessageSender_h
#define FWCore_MessageLogger_MessageSender_h

#include "FWCore/MessageLogger/interface/ELstring.h"
#include "FWCore/MessageLogger/interface/ELseverityLevel.h"
#include "FWCore/MessageLogger/interface/ErrorObj.h"

#include <memory>

#include <map>
#include "FWCore/Utilities/interface/HideStdSharedPtrFromRoot.h"

// Change log
//
//  1  mf 8/25/08	error summary information for LoggedErrorsSummary()
//
//  2  mf 6/22/09	add severity to LoggedErrorsSummary by using 
//			ErrorSummaryEntry as map key
//			
//  3 wmtan 6/22/11     Hold the ErrorObj with a shared pointer with a custom deleter.
//                      The custom deleter takes over the function of the message sending from the MessageSender destructor.
//                      This allows MessageSender to be copyable, which fixes the clang compilation errors.
         

namespace edm
{

class MessageSender
{
  struct ErrorObjDeleter {
    ErrorObjDeleter() {}
    void operator()(ErrorObj * errorObjPtr);
  };

public:
  // ---  birth/death:
  MessageSender() : errorobj_p() {} 
  MessageSender( ELseverityLevel const & sev, 
  		 ELstring const & id,
		 bool verbatim = false, bool suppressed = false );
  ~MessageSender();

  // ---  stream out the next part of a message:
  template< class T >
    MessageSender &
    operator<< ( T const & t )
  {
#ifndef __GCCXML__
    if (valid()) (*errorobj_p) << t;
#endif
    return *this;
  }

  bool valid() {
    return errorobj_p != 0;
  }
  
private:
  // data:
  std::shared_ptr<ErrorObj> errorobj_p;

};  // MessageSender


}  // namespace edm


#endif  // FWCore_MessageLogger_MessageSender_h
