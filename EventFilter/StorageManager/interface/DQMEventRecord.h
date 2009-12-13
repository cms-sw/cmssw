// $Id: DQMEventRecord.h,v 1.5 2009/09/16 11:04:22 mommsen Exp $
/// @file: DQMEventRecord.h 

#ifndef StorageManager_DQMEventRecord_h
#define StorageManager_DQMEventRecord_h

#include <vector>

#include "boost/shared_ptr.hpp"

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/DQMInstance.h"
#include "EventFilter/StorageManager/interface/DQMKey.h"

#include "IOPool/Streamer/interface/DQMEventMessage.h"


namespace stor {

  class DQMEventMonitorCollection;
  class QueueID;


  /**
   * Class holding information for one DQM event
   *
   * $Author: mommsen $
   * $Revision: 1.5 $
   * $Date: 2009/09/16 11:04:22 $
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
      bool empty() const
      { return ( _entry->buffer.empty() ); }

      /**
       * Returns the size of the stored event msg view in bytes
       */
      size_t totalDataSize() const
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
    unsigned int _updateCount; //incremented for each new event being added.
                               //Note that nUpdates_ is incremented for each folder
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
