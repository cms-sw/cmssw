#ifndef FWCore_MessageService_SingleThreadMSPresence_h
#define FWCore_MessageService_SingleThreadMSPresence_h

#include "FWCore/Utilities/interface/Presence.h"

namespace edm  {
namespace service {       

class SingleThreadMSPresence : public Presence
{
public:
  // ---  birth/death:
  SingleThreadMSPresence();
  ~SingleThreadMSPresence();

  // --- Access to the scribe
  // REMOVED AbstractMLscribe * scribe_ptr() { return &m; }  

private:
  // --- no copying:
  SingleThreadMSPresence(SingleThreadMSPresence const &);
  void  operator = (SingleThreadMSPresence const &);
  
};  // SingleThreadMSPresence


}   // end of namespace service
}  // namespace edm


#endif  // FWCore_MessageService_SingleThreadMSPresence_h
