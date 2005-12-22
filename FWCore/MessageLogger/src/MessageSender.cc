#include "FWCore/MessageLogger/interface/MessageSender.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <iostream>


using namespace edm;


MessageSender::MessageSender( ELseverityLevel const & sev, ELstring const & id )
: errorobj_p( new ErrorObj(sev,id) )
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

  try
    {
      edm::Service<edm::service::MessageLogger> h;
      h->fillErrorObj(*errorobj_p);
    }
  catch(cms::Exception&)
    {
      // we get here because no services are available yet - ignore the error
      errorobj_p->setModule("pre-services");
    }
  catch(...)
    {
      errorobj_p->setModule("Service:unknown exception occurred");
    }
  
  MessageLoggerQ::LOG(errorobj_p);
}
