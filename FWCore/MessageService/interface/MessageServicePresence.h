#ifndef FWCore_MessageService_MessageServicePresence_h
#define FWCore_MessageService_MessageServicePresence_h

#include "FWCore/Utilities/interface/Presence.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

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
  ~MessageServicePresence() override;

private:
  // --- no copying:
  MessageServicePresence(MessageServicePresence const&) = delete; // Disallow copying
  void operator=(MessageServicePresence const &) = delete; // Disallow copying

  std::shared_ptr<ThreadQueue const> queue() const {return get_underlying_safe(m_queue);}
  std::shared_ptr<ThreadQueue>& queue() {return get_underlying_safe(m_queue);}

  // --- data:
  edm::propagate_const<std::shared_ptr<ThreadQueue>> m_queue;
  boost::thread  m_scribeThread;

};  // MessageServicePresence


}   // end of namespace service
}  // namespace edm


#endif  // FWCore_MessageService_MessageServicePresence_h
