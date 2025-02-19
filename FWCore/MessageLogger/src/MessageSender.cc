#include "FWCore/MessageLogger/interface/MessageSender.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#define TRACE_DROP
#ifdef TRACE_DROP
#include <iostream>
#endif

// Change log
//
//  1  mf 8/25/08	keeping the error summary information for
//			LoggedErrorsSummary()
//			
// 2  mf 11/2/10	Use new moduleContext method of MessageDrop:
//			see MessageServer/src/MessageLogger.cc change 17.
//			


using namespace edm;

bool MessageSender::errorSummaryIsBeingKept = false;		// change log 1
bool MessageSender::freshError              = false;
std::map<ErrorSummaryMapKey, unsigned int> MessageSender::errorSummaryMap; 

MessageSender::MessageSender( ELseverityLevel const & sev, 
			      ELstring const & id,
			      bool verbatim, bool suppressed )
: errorobj_p( suppressed ? 0 : new ErrorObj(sev,id,verbatim), ErrorObjDeleter())
{
  //std::cout << "MessageSender ctor; new ErrorObj at: " << errorobj_p << '\n';
}


// This destructor must not be permitted to throw. A
// boost::thread_resoruce_error is thrown at static destruction time,
// if the MessageLogger library is loaded -- even if it is not used.
void MessageSender::ErrorObjDeleter::operator()(ErrorObj * errorObjPtr) {
  if (errorObjPtr == 0) {
    return;
  }
  try 
    {
      //std::cout << "MessageSender dtor; ErrorObj at: " << errorobj_p << '\n';

      // surrender ownership of our ErrorObj, transferring ownership
      // (via the intermediate MessageLoggerQ) to the MessageLoggerScribe
      // that will (a) route the message text to its destination(s)
      // and will then (b) dispose of the ErrorObj
      
      MessageDrop * drop = MessageDrop::instance();
      if (drop) {
	errorObjPtr->setModule(drop->moduleContext());		// change log 
	errorObjPtr->setContext(drop->runEvent);
      } 
#ifdef TRACE_DROP
      if (!drop) std::cerr << "MessageSender::~MessageSender() - Null drop pointer \n";
#endif
								// change log 1
      if ( errorSummaryIsBeingKept && 
           errorObjPtr->xid().severity >= ELwarning ) 
      {				
	ELextendedID const & xid = errorObjPtr->xid();
        ErrorSummaryMapKey key (xid.id, xid.module, xid.severity);
	ErrorSummaryMapIterator i = errorSummaryMap.find(key);
	if (i != errorSummaryMap.end()) {
	  ++(i->second);  // same as ++errorSummaryMap[key]
	} else {
	  errorSummaryMap[key] = 1;
	}
	freshError = true;
      }
      
      MessageLoggerQ::MLqLOG(errorObjPtr);
    }
  catch ( ... )
    {
      // nothing to do
      
      // for test that removal of thread-involved static works, 
      // simply throw here, then run in trivial_main in totalview
      // and Next or Step so that the exception would be detected.
      // That test has been done 12/14/07.
    }
}
MessageSender::~MessageSender()
{
}
