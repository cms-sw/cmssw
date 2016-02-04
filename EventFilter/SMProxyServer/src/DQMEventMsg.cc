// $Id: DQMEventMsg.cc,v 1.2 2011/03/07 15:41:54 mommsen Exp $
/// @file: DQMEventMsg.cc

#include "EventFilter/SMProxyServer/interface/DQMEventMsg.h"


namespace smproxy
{

  DQMEventMsg::DQMEventMsg() :
  faulty_(true)
  {}
  
  
  DQMEventMsg::DQMEventMsg(const DQMEventMsgView& dqmEventMsgView) :
  faulty_(false)
  {
    buf_.reset( new DQMEventMsgBuffer(dqmEventMsgView.size()) );
    std::copy(
      dqmEventMsgView.startAddress(),
      dqmEventMsgView.startAddress()+dqmEventMsgView.size(),
      &(*buf_)[0]
    );
    dqmKey_.runNumber = dqmEventMsgView.runNumber();
    dqmKey_.lumiSection = dqmEventMsgView.lumiSection();
    dqmKey_.topLevelFolderName = dqmEventMsgView.topFolderName();
  }
  
  
  void DQMEventMsg::tagForDQMEventConsumers(const stor::QueueIDs& queueIDs)
  {
    queueIDs_ = queueIDs;
  }
  
  
  const stor::QueueIDs& DQMEventMsg::getDQMEventConsumerTags() const
  {
    return queueIDs_;
  }
  
  
  const stor::DQMKey& DQMEventMsg::dqmKey() const
  {
    return dqmKey_;
  }

  
  size_t DQMEventMsg::memoryUsed() const
  {
    return sizeof(buf_);
  }
  
  
  unsigned long DQMEventMsg::totalDataSize() const
  {
    return buf_->size();
  }
  
  
  unsigned char* DQMEventMsg::dataLocation() const
  {
    return &(*buf_)[0];
  }
  
  
  bool DQMEventMsg::empty() const
  {
    return (buf_.get() == 0);
  }


  bool DQMEventMsg::faulty() const
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
