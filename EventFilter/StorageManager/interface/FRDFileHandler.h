// $Id: FRDFileHandler.h,v 1.3 2009/07/03 11:08:32 mommsen Exp $
/// @file: FRDFileHandler.h 

#ifndef StorageManager_FRDFileHandler_h
#define StorageManager_FRDFileHandler_h

#include <EventFilter/StorageManager/interface/FileHandler.h>
#include <IOPool/Streamer/interface/FRDEventFileWriter.h>

namespace stor {
  
  /**
   * Represents a file holding HLT error events in the
   * FED Raw Data (FRD) format.
   *
   * $Author: mommsen $
   * $Revision: 1.3 $
   * $Date: 2009/07/03 11:08:32 $
   */
  
  class FRDFileHandler : public FileHandler
  {
  public:
    FRDFileHandler
    (
      FilesMonitorCollection::FileRecordPtr,
      const DiskWritingParams&,
      const long long& maxFileSize
    );
    
    virtual ~FRDFileHandler();
        
    /**
     * Write the event contained in the I2OChain
     */
    virtual void writeEvent(const I2OChain&);

    /**
     *  Returns true if the file has not seen any recent events
     */
    virtual const bool tooOld(utils::time_point_t currentTime = utils::getCurrentTime());

    
  private:
    
    /**
     * Close the file
     */
    virtual void closeFile();
    
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
