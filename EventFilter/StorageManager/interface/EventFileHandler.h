// $Id: EventFileHandler.h,v 1.15 2012/04/04 12:16:56 mommsen Exp $
/// @file: EventFileHandler.h 

#ifndef EventFilter_StorageManager_EventFileHandler_h
#define EventFilter_StorageManager_EventFileHandler_h

#include "EventFilter/StorageManager/interface/FileHandler.h"
#include "EventFilter/StorageManager/interface/InitMsgCollection.h"

#include "IOPool/Streamer/src/StreamerFileWriter.h"

#include <stdint.h>
#include <boost/scoped_ptr.hpp>


namespace stor {

  class I2OChain;

  
  /**
   * Represents a file holding event data
   *
   * $Author: mommsen $
   * $Revision: 1.15 $
   * $Date: 2012/04/04 12:16:56 $
   */
  
  class EventFileHandler : public FileHandler
  {
  public:
    EventFileHandler
    (
      InitMsgSharedPtr,
      FilesMonitorCollection::FileRecordPtr,
      const DbFileHandlerPtr,
      const uint64_t& maxFileSize
    );

    /**
     * Close the file
     */
    virtual void closeFile(const FilesMonitorCollection::FileRecord::ClosingReason&);


  private:
    
    /**
     * Write the init message to the head of the file
     */
    void writeHeader(InitMsgSharedPtr);

    /**
     * Write the I2OChain to the file
     */
    virtual void do_writeEvent(const I2OChain&);

    boost::scoped_ptr<edm::StreamerFileWriter> writer_; // writes streamer and index file
  };
  
} // stor namespace

#endif // EventFilter_StorageManager_EventFileHandler_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
