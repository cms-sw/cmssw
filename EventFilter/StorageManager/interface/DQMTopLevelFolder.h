// $Id: DQMTopLevelFolder.h,v 1.5 2011/04/04 16:05:37 mommsen Exp $
/// @file: DQMTopLevelFolder.h 

#ifndef EventFilter_StorageManager_DQMTopLevelFolder_h
#define EventFilter_StorageManager_DQMTopLevelFolder_h

#include <vector>

#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/Timestamp.h"

#include "EventFilter/StorageManager/interface/AlarmHandler.h"
#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/DQMFolder.h"
#include "EventFilter/StorageManager/interface/DQMKey.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/QueueID.h"

#include "IOPool/Streamer/interface/DQMEventMessage.h"


namespace stor {

  class DQMEventMonitorCollection;


  /**
   * Class holding information for one DQM event
   *
   * $Author: mommsen $
   * $Revision: 1.5 $
   * $Date: 2011/04/04 16:05:37 $
   */

  class DQMTopLevelFolder
  {

  public:

    class Record
    {
    private:

      struct Entry
      {
        std::vector<unsigned char> buffer;
        QueueIDs dqmConsumers;
      };

    public:
      
      Record() :
      entry_(new Entry) {};

      /**
       * Clear any data
       */
      inline void clear()
      { entry_->buffer.clear(); entry_->dqmConsumers.clear(); }

      /**
       * Return a reference to the buffer providing space for
       * the specified size in bytes.
       */
      inline void* getBuffer(size_t size) const
      { entry_->buffer.resize(size); return &(entry_->buffer[0]); }

      /**
       * Set the list of DQM event consumer this
       * top level folder should be served to.
       */
      inline void tagForEventConsumers(const QueueIDs& ids)
      { entry_->dqmConsumers = ids; }
      
      /**
       * Get the list of DQM event consumers this
       * top level folder should be served to.
       */
      inline QueueIDs getEventConsumerTags() const
       { return entry_->dqmConsumers; }

      /**
       * Returns the DQM event message view for this group
       */
      inline DQMEventMsgView getDQMEventMsgView() const
      { return DQMEventMsgView(&entry_->buffer[0]); }

      /**
       * Returns true if there is no DQM event message view available
       */
      inline bool empty() const
      { return ( entry_->buffer.empty() ); }

      /**
       * Returns the memory usage of the stored event msg view in bytes
       */
      inline size_t memoryUsed() const
      { return entry_->buffer.size() + entry_->dqmConsumers.size()*sizeof(QueueID); }

      /**
       * Returns the size of the stored event msg view in bytes
       */
      inline unsigned long totalDataSize() const
      { return entry_->buffer.size(); }


    private:

      // We use here a shared_ptr to avoid copying the whole
      // buffer each time the event record is handed on
      boost::shared_ptr<Entry> entry_;
      
    };
    
    
  public:

    DQMTopLevelFolder
    (
      const DQMKey&,
      const QueueIDs&,
      const DQMProcessingParams&,
      DQMEventMonitorCollection&,
      const unsigned int expectedUpdates,
      AlarmHandlerPtr
    );

    ~DQMTopLevelFolder();

    /**
     * Adds the DQMEventMsgView, but does not take ownership of the underlying
     * data buffer. Collates the histograms with the existing
     * DQMEventMsgView if there is one.
     */
    void addDQMEvent(const DQMEventMsgView&);

    /**
     * Adds the DQM event message contained in the I2OChain.
     * It copies the data from the I2OChain in its own buffer space.
     * Collates the histograms with the existing
     * DQMEventMsgView if there is one.
     */
    void addDQMEvent(const I2OChain& dqmEvent);

    /**
     * Returns true if this top level folder is ready to be served.
     * This is either the case if all expected updates have been received
     * or when the last update was more than dqmParams.readyTimeDQM ago.
     */
    bool isReady(const utils::TimePoint_t& now) const;

    /**
     * Populate the record with the currently available data.
     * Return false if no data is available.
     */
    bool getRecord(Record&);


  private:

    void addEvent(std::auto_ptr<DQMEvent::TObjectTable>);
    size_t populateTable(DQMEvent::TObjectTable&) const;

    const DQMKey dqmKey_;
    const QueueIDs dqmConsumers_;
    const DQMProcessingParams dqmParams_;
    DQMEventMonitorCollection& dqmEventMonColl_;
    const unsigned int expectedUpdates_;
    AlarmHandlerPtr alarmHandler_;

    unsigned int nUpdates_;
    unsigned int mergeCount_;
    utils::TimePoint_t lastUpdate_;
    std::string releaseTag_;
    uint32_t updateNumber_;
    edm::Timestamp timeStamp_;

    typedef boost::shared_ptr<DQMFolder> DQMFolderPtr;
    typedef std::map<std::string, DQMFolderPtr> DQMFoldersMap;
    DQMFoldersMap dqmFolders_;
    
    static unsigned int sentEvents_;    
  };

  typedef boost::shared_ptr<DQMTopLevelFolder> DQMTopLevelFolderPtr;

} // namespace stor

#endif // EventFilter_StorageManager_DQMTopLevelFolder_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
