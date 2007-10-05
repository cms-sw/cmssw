#include "FWCore/MessageLogger/interface/MessageSender.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#define TRACE_DROP
#ifdef TRACE_DROP
#include <iostream>
#endif

using namespace edm;


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


MessageSender::~MessageSender()
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

  MessageLoggerQ::MLqLOG(errorobj_p);
}
