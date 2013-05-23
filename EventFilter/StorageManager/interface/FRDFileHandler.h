// $Id: FRDFileHandler.h,v 1.12.4.1 2011/03/07 11:33:04 mommsen Exp $
/// @file: FRDFileHandler.h 

#ifndef EventFilter_StorageManager_FRDFileHandler_h
#define EventFilter_StorageManager_FRDFileHandler_h

#include "EventFilter/StorageManager/interface/FileHandler.h"
#include "IOPool/Streamer/interface/FRDEventFileWriter.h"

#include <stdint.h>
#include <boost/scoped_ptr.hpp>


namespace stor {

  class I2OChain;

  
  /**
   * Represents a file holding HLT error events in the
   * FED Raw Data (FRD) format.
   *
   * $Author: mommsen $
   * $Revision: 1.12.4.1 $
   * $Date: 2011/03/07 11:33:04 $
   */
  
  class FRDFileHandler : public FileHandler
  {
  public:
    FRDFileHandler
    (
      FilesMonitorCollection::FileRecordPtr,
      const DbFileHandlerPtr,
      const DiskWritingParams&,
      const uint64_t& maxFileSize
    );

    /**
     * Close the file
     */
    virtual void closeFile(const FilesMonitorCollection::FileRecord::ClosingReason&);


  private:
    
    /**
     * Write the I2OChain to the file
     */
    virtual void do_writeEvent(const I2OChain&);

    boost::scoped_ptr<FRDEventFileWriter> writer_; // writes FED Raw Data file
  };
  
} // stor namespace

#endif // EventFilter_StorageManager_FRDFileHandler_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
