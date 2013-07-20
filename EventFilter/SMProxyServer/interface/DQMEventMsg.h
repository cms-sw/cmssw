// $Id: DQMEventMsg.h,v 1.2 2011/03/07 15:41:54 mommsen Exp $
/// @file: DQMEventMsg.h 

#ifndef EventFilter_SMProxyServer_DQMEventMsg_h
#define EventFilter_SMProxyServer_DQMEventMsg_h

#include "EventFilter/StorageManager/interface/DQMKey.h"
#include "EventFilter/StorageManager/interface/QueueID.h"
#include "IOPool/Streamer/interface/DQMEventMessage.h"

#include <boost/shared_ptr.hpp>
#include <vector>

namespace smproxy {

  /**
   * A class for storing an DQMEventMsgView and providing the basic 
   * APIs required for SMPS
   *
   * $Author: mommsen $
   * $Revision: 1.2 $
   * $Date: 2011/03/07 15:41:54 $
   */

  class DQMEventMsg
  {
  public:
    
    DQMEventMsg();
    DQMEventMsg(const DQMEventMsgView&);

    /**
      Tag the DQM event for the passed list of queueIDs
     */
    void tagForDQMEventConsumers(const stor::QueueIDs&);

    /**
      Return the QueueIDs for which the DQM event is tagged
     */
    const stor::QueueIDs& getDQMEventConsumerTags() const;

    /**
      Return the DQM key corresponding to this message
    */
    const stor::DQMKey& dqmKey() const;

    /**
      Returns the total memory occupied by the event message
     */
    size_t memoryUsed() const;
    
    /**
      Returns the size of the event message
     */
    unsigned long totalDataSize() const;

    /**
      Returns the start adderess of the event message
     */
    unsigned char* dataLocation() const;

    /**
      Returns true if no event message is managed by *this
     */
    bool empty() const;

    /**
      Returns true if the event message is faulty
     */
    bool faulty() const;


  private:
    typedef std::vector<unsigned char> DQMEventMsgBuffer;
    boost::shared_ptr<DQMEventMsgBuffer> buf_;
    bool faulty_;

    stor::QueueIDs queueIDs_;
    stor::DQMKey dqmKey_;
  };
  
} // namespace smproxy

#endif // EventFilter_SMProxyServer_DQMEventMsg_h 



// emacs configuration
// Local Variables: -
// mode: c++ -
// c-basic-offset: 2 -
// indent-tabs-mode: nil -
// End: -
