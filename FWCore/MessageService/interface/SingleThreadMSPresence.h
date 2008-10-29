#ifndef FWCore_MessageService_SingleThreadMSPresence_h
#define FWCore_MessageService_SingleThreadMSPresence_h

#include "boost/thread/thread.hpp"
#include "FWCore/Utilities/interface/Presence.h"
#include "FWCore/MessageService/interface/MessageLoggerScribe.h"
#include "FWCore/MessageLogger/interface/AbstractMLscribe.h"

namespace edm  {
namespace service {       

class SingleThreadMSPresence : public Presence
{
public:
  // ---  birth/death:
  SingleThreadMSPresence();
  ~SingleThreadMSPresence();

  // --- Access to the scribe
  AbstractMLscribe * scribe_ptr() { return &m; }  

private:
  // --- no copying:
  SingleThreadMSPresence(SingleThreadMSPresence const &);
  void  operator = (SingleThreadMSPresence const &);
  MessageLoggerScribe m; 
  
};  // SingleThreadMSPresence


}   // end of namespace service
}  // namespace edm


#endif  // FWCore_MessageService_SingleThreadMSPresence_h
