// $Id: FRDFileHandler.h,v 1.11 2010/09/09 08:01:16 mommsen Exp $
/// @file: FRDFileHandler.h 

#ifndef StorageManager_FRDFileHandler_h
#define StorageManager_FRDFileHandler_h

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
   * $Revision: 1.11 $
   * $Date: 2010/09/09 08:01:16 $
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

    boost::scoped_ptr<FRDEventFileWriter> _writer; // writes FED Raw Data file
  };
  
} // stor namespace

#endif // StorageManager_FRDFileHandler_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
