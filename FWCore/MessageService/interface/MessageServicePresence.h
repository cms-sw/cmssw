#ifndef FWCore_MessageService_MessageServicePresence_h
#define FWCore_MessageService_MessageServicePresence_h

#include "FWCore/Utilities/interface/Presence.h"

#include "boost/thread/thread.hpp"

#include <memory>


namespace edm  {
namespace service {       

class ThreadQueue;

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
  std::shared_ptr<ThreadQueue> m_queue;
  boost::thread  m_scribeThread;

};  // MessageServicePresence


}   // end of namespace service
}  // namespace edm


#endif  // FWCore_MessageService_MessageServicePresence_h
