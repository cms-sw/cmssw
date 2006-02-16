#ifndef FWCore_MessageService_MessageServicePresence_h
#define FWCore_MessageService_MessageServicePresence_h


#include "boost/thread/thread.hpp"
#include "FWCore/Utilities/interface/Presence.h"


namespace edm  {
namespace service {       


class MessageServicePresence : public Presence
{
public:
  // ---  birth/death:
  MessageServicePresence();
  ~MessageServicePresence();

private:
  // --- no copying:
  MessageServicePresence(MessageServicePresence const &);
  void  operator = (MessageServicePresence const &);

  // --- data:
  boost::thread  scribe;

};  // MessageServicePresence


}   // end of namespace service
}  // namespace edm


#endif  // FWCore_MessageService_MessageServicePresence_h
