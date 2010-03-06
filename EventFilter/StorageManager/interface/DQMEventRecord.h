// $Id: DQMEventRecord.h,v 1.9 2010/03/04 16:58:35 mommsen Exp $
/// @file: DQMEventRecord.h 

#ifndef StorageManager_DQMEventRecord_h
#define StorageManager_DQMEventRecord_h

#include <vector>

#include "boost/shared_ptr.hpp"

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/DQMInstance.h"
#include "EventFilter/StorageManager/interface/DQMKey.h"
#include "EventFilter/StorageManager/interface/QueueID.h"

#include "IOPool/Streamer/interface/DQMEventMessage.h"


namespace stor {

  class DQMEventMonitorCollection;


  /**
   * Class holding information for one DQM event
   *
   * $Author: mommsen $
   * $Revision: 1.9 $
   * $Date: 2010/03/04 16:58:35 $
   */

  class DQMEventRecord : public DQMInstance
  {

  public:

    struct GroupRecord
    {
      struct Entry
      {
        std::vector<unsigned char> buffer;
        std::vector<QueueID> dqmConsumers;
      };
      
      GroupRecord() :
      _entry(new Entry) {};
      
      /**
       * Get the list of DQM event consumers this
       * DQM event group should be served to.
       */
      std::vector<QueueID> getEventConsumerTags() const
       { return _entry->dqmConsumers; }

      /**
       * Returns the DQM event message view for this group
       */
      DQMEventMsgView getDQMEventMsgView()
      { return DQMEventMsgView(&_entry->buffer[0]); }

      /**
       * Returns true if there is no DQM event message view available
       */
      inline bool empty() const
      { return ( _entry->buffer.empty() ); }

      /**
       * Returns the memory usage of the stored event msg view in bytes
       */
      inline size_t memoryUsed() const
      { return _entry->buffer.size() + _entry->dqmConsumers.size()*sizeof(QueueID); }

      /**
       * Returns the size of the stored event msg view in bytes
       */
      inline unsigned long totalDataSize() const
      { return _entry->buffer.size(); }

      // We use here a shared_ptr to avoid copying the whole
      // buffer each time the event record is handed on
      boost::shared_ptr<Entry> _entry;
      
    };
    
    
  public:

    DQMEventRecord
    (
      const DQMKey,
      const DQMProcessingParams,
      DQMEventMonitorCollection&,
      const unsigned int expectedUpdates
    );

    ~DQMEventRecord();

    /**
     * Set the list of DQM event consumers this
     * DQM event should be served to.
     */
    void setEventConsumerTags(std::vector<QueueID> dqmConsumers)
    { _dqmConsumers = dqmConsumers; }

    /**
     * Adds the DQMEventMsgView. Collates the histograms with the existing
     * DQMEventMsgView if there is one.
     */
    void addDQMEventView(DQMEventMsgView const&);

    /**
     * Writes the histograms hold to file
     */ 
    double writeFile(std::string filePrefix, bool endRunFlag);

    /**
     * Populates the dqmEventView with the requested group and returns the group
     */
    GroupRecord populateAndGetGroup(const std::string groupName);


  private:

    const DQMProcessingParams _dqmParams;
    DQMEventMonitorCollection& _dqmEventMonColl;

    std::vector<QueueID> _dqmConsumers;
    std::string _releaseTag;

    unsigned int _sentEvents;
  };

  typedef boost::shared_ptr<DQMEventRecord> DQMEventRecordPtr;

} // namespace stor

#endif // StorageManager_DQMEventRecord_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
