// $Id: EventMsg.cc,v 1.3 2011/03/08 18:44:55 mommsen Exp $
/// @file: EventMsg.cc

#include "EventFilter/SMProxyServer/interface/EventMsg.h"


namespace smproxy
{

  EventMsg::EventMsg() :
  faulty_(true),
  droppedEventsCount_(0)
  {}
  
  
  EventMsg::EventMsg(const EventMsgView& eventMsgView) :
  faulty_(false)
  {
    buf_.reset( new EventMsgBuffer(eventMsgView.size()) );
    std::copy(
      eventMsgView.startAddress(),
      eventMsgView.startAddress()+eventMsgView.size(),
      &(*buf_)[0]
    );
    droppedEventsCount_ = eventMsgView.droppedEventsCount();
  }
  
  
  void EventMsg::tagForEventConsumers(const stor::QueueIDs& queueIDs)
  {
    queueIDs_ = queueIDs;
  }
  
  
  const stor::QueueIDs& EventMsg::getEventConsumerTags() const
  {
    return queueIDs_;
  }
  
  
  unsigned int EventMsg::droppedEventsCount() const
  {
    return droppedEventsCount_;
  }
  
  
  void EventMsg::setDroppedEventsCount(unsigned int count)
  {
    EventHeader* header = (EventHeader*)dataLocation();
    convert(count,header->droppedEventsCount_);
  }
  
  
  size_t EventMsg::memoryUsed() const
  {
    return sizeof(buf_);
  }
  
  
  unsigned long EventMsg::totalDataSize() const
  {
    return buf_->size();
  }
  
  
  unsigned char* EventMsg::dataLocation() const
  {
    return &(*buf_)[0];
  }
  
  
  bool EventMsg::empty() const
  {
    return (buf_.get() == 0);
  }


  bool EventMsg::faulty() const
  {
    return faulty_;
  }

} // namespace smproxy
  
/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
