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


using namespace edm;

bool MessageSender::errorSummaryIsBeingKept = false;		// change log 1
bool MessageSender::freshError              = false;
std::map<ErrorSummaryMapKey, unsigned int> MessageSender::errorSummaryMap; 

MessageSender::MessageSender( ELseverityLevel const & sev, 
			      ELstring const & id,
			      bool verbatim )
: errorobj_p( new ErrorObj(sev,id,verbatim) )
{
  //std::cout << "MessageSender ctor; new ErrorObj at: " << errorobj_p << '\n';
}

MessageSender::MessageSender( ELseverityLevel const & sev, 
			      ELstring const & id )
: errorobj_p( new ErrorObj(sev,id,false) )
{
  //std::cout << "MessageSender ctor; new ErrorObj at: " << errorobj_p << '\n';
}


// This destructor must not be permitted to throw. A
// boost::thread_resoruce_error is thrown at static destruction time,
// if the MessageLogger library is loaded -- even if it is not used.
MessageSender::~MessageSender()
{
  try 
    {
      //std::cout << "MessageSender dtor; ErrorObj at: " << errorobj_p << '\n';

      // surrender ownership of our ErrorObj, transferring ownership
      // (via the intermediate MessageLoggerQ) to the MessageLoggerScribe
      // that will (a) route the message text to its destination(s)
      // and will then (b) dispose of the ErrorObj
      
      MessageDrop * drop = MessageDrop::instance();
      if (drop) {
	errorobj_p->setModule(drop->moduleName);
	errorobj_p->setContext(drop->runEvent);
      } 
#ifdef TRACE_DROP
      if (!drop) std::cerr << "MessageSender::~MessageSender() - Null drop pointer \n";
#endif
								// change log 1
      if ( errorSummaryIsBeingKept && 
           errorobj_p->xid().severity >= ELwarning ) 
      {				
	ELextendedID const & xid = errorobj_p->xid();
        ErrorSummaryMapKey key (xid.id, xid.module);
	ErrorSummaryMapIterator i = errorSummaryMap.find(key);
	if (i != errorSummaryMap.end()) {
	  ++(i->second);  // same as ++errorSummaryMap[key]
	} else {
	  errorSummaryMap[key] = 1;
	}
	freshError = true;
      }
      
      MessageLoggerQ::MLqLOG(errorobj_p);
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
