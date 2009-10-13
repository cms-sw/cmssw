// $Id: FRDFileHandler.h,v 1.6 2009/09/17 11:04:43 mommsen Exp $
/// @file: FRDFileHandler.h 

#ifndef StorageManager_FRDFileHandler_h
#define StorageManager_FRDFileHandler_h

#include "EventFilter/StorageManager/interface/FileHandler.h"
#include "IOPool/Streamer/interface/FRDEventFileWriter.h"

#include <stdint.h>

namespace stor {
  
  /**
   * Represents a file holding HLT error events in the
   * FED Raw Data (FRD) format.
   *
   * $Author: mommsen $
   * $Revision: 1.6 $
   * $Date: 2009/09/17 11:04:43 $
   */
  
  class FRDFileHandler : public FileHandler
  {
  public:
    FRDFileHandler
    (
      FilesMonitorCollection::FileRecordPtr,
      const DiskWritingParams&,
      const unsigned long long& maxFileSize
    );
    
    /**
     * Write the event contained in the I2OChain
     */
    virtual void writeEvent(const I2OChain&);

    /**
     * Returns true if the file has not seen any recent events
     */
    virtual bool tooOld(utils::time_point_t currentTime = utils::getCurrentTime());

    /**
     * Error events do not belong to a lumi section
     */
    virtual bool isFromLumiSection(const uint32_t lumiSection)
    { return false; }

    /**
     * Close the file
     */
    virtual void closeFile(const FilesMonitorCollection::FileRecord::ClosingReason&);
    
  private:
    
    FRDEventFileWriter _writer;
  };
  
} // stor namespace

#endif // StorageManager_FRDFileHandler_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
