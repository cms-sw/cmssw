// $Id: EventFileHandler.h,v 1.5 2009/08/28 16:41:49 mommsen Exp $
/// @file: EventFileHandler.h 

#ifndef StorageManager_EventFileHandler_h
#define StorageManager_EventFileHandler_h

#include "EventFilter/StorageManager/interface/FileHandler.h"
#include "EventFilter/StorageManager/interface/InitMsgCollection.h"

#include <IOPool/Streamer/src/StreamerFileWriter.h>


namespace stor {

  class I2OChain;

  
  /**
   * Represents a file holding event data
   *
   * $Author: mommsen $
   * $Revision: 1.5 $
   * $Date: 2009/08/28 16:41:49 $
   */
  
  class EventFileHandler : public FileHandler
  {
  public:
    EventFileHandler
    (
      InitMsgSharedPtr,
      FilesMonitorCollection::FileRecordPtr,
      const DiskWritingParams&,
      const long long& maxFileSize
    );

    virtual ~EventFileHandler();
        
    /**
     * Write the event contained in the I2OChain
     */
    virtual void writeEvent(const I2OChain&);
 
    /**
     *  Returns true if the file has not seen any recent events
     */
    virtual bool tooOld(const utils::time_point_t currentTime = utils::getCurrentTime());
  
    /**
     * Returns true if the file corresponds to the given lumi section
     */
    virtual bool isFromLumiSection(const uint32_t lumiSection);


  private:
    
    /**
     * Close the file
     */
    virtual void closeFile();

    /**
     * Write the init message to the head of the file
     */
    void writeHeader(InitMsgSharedPtr);
    

    edm::StreamerFileWriter _writer; // writes streamer and index file
  };
  
} // stor namespace

#endif // StorageManager_EventFileHandler_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
