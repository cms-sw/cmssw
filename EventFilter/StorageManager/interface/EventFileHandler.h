// $Id: EventFileHandler.h,v 1.9 2010/02/01 11:39:46 mommsen Exp $
/// @file: EventFileHandler.h 

#ifndef StorageManager_EventFileHandler_h
#define StorageManager_EventFileHandler_h

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
   * $Revision: 1.9 $
   * $Date: 2010/02/01 11:39:46 $
   */
  
  class EventFileHandler : public FileHandler
  {
  public:
    EventFileHandler
    (
      InitMsgSharedPtr,
      FilesMonitorCollection::FileRecordPtr,
      const DiskWritingParams&,
      const unsigned long long& maxFileSize
    );
 
    /**
     *  Returns true if the file has not seen any recent events
     */
    virtual bool tooOld(const utils::time_point_t currentTime = utils::getCurrentTime());
  
    /**
     * Returns true if the file corresponds to the given lumi section
     */
    virtual bool isFromLumiSection(const uint32_t lumiSection);

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

    boost::scoped_ptr<edm::StreamerFileWriter> _writer; // writes streamer and index file
  };
  
} // stor namespace

#endif // StorageManager_EventFileHandler_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
