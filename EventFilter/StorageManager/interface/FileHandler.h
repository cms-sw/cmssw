// $Id: FileHandler.h,v 1.21 2012/04/04 12:16:57 mommsen Exp $
/// @file: FileHandler.h 

#ifndef EventFilter_StorageManager_FileHandler_h
#define EventFilter_StorageManager_FileHandler_h

#include "EventFilter/StorageManager/interface/Configuration.h"
#include "EventFilter/StorageManager/interface/DbFileHandler.h"
#include "EventFilter/StorageManager/interface/FilesMonitorCollection.h"
#include "EventFilter/StorageManager/interface/Utils.h"
#include "IOPool/Streamer/interface/MsgHeader.h"

#include "boost/shared_ptr.hpp"

#include <stdint.h>
#include <string>
#include <sys/types.h>


namespace stor {

  class I2OChain;

  /**
   * Abstract representation of a physical file
   *
   * $Author: mommsen $
   * $Revision: 1.21 $
   * $Date: 2012/04/04 12:16:57 $
   */

  class FileHandler
  {
  public:
        
    FileHandler
    (
      FilesMonitorCollection::FileRecordPtr,
      const DbFileHandlerPtr,
      const uint64_t& maxFileSize
     );
    
    virtual ~FileHandler() {};
 
    /**
     * Write the event contained in the I2OChain
     */
    void writeEvent(const I2OChain&);

    /**
     * Returns true if the file has not seen any recent events
     */
    bool tooOld(const utils::TimePoint_t currentTime = utils::getCurrentTime());

    /**
     * Returns true if the file corresponds to the given lumi section
     */
    bool isFromLumiSection(const uint32_t lumiSection);

    /**
     * Returns true if the additional data size would push the file size
     * beyond maxFileSize.
     */
    bool tooLarge(const uint64_t& dataSize);


    ////////////////////////////
    // File parameter getters //
    ////////////////////////////
    
    /**
     * Return the number of events in the file
     */
    uint32_t events() const;
    
    /**
     * Return the luminosity section the file belongs to
     */
    uint32_t lumiSection() const
    { return fileRecord_->lumiSection; }
    
    /**
     * Return the size of the file in bytes
     */
    uint64_t fileSize() const;

    /**
     * Close the file
     */
    virtual void closeFile(const FilesMonitorCollection::FileRecord::ClosingReason&) = 0;


  protected:

    /**
     * Write the I2OChain to the file
     */
    virtual void do_writeEvent(const I2OChain& event) = 0;
    
    
    //////////////////////
    // File bookkeeping //
    //////////////////////

    /**
     * Write command to update the file information in the CMS_STOMGR.TIER0_INJECTION table
     * into the logFile_.
     */
    void updateDatabase() const;


    /**
     * Write command to insert a new file into the CMS_STOMGR.TIER0_INJECTION table
     * into the logFile_.
     */
    void insertFileInDatabase() const;

    
    /////////////////////////////
    // File system interaction //
    /////////////////////////////
    
    /**
     * Move index and streamer file to "closed" directory
     */
    void moveFileToClosed
    (
      const FilesMonitorCollection::FileRecord::ClosingReason&
    );


  private:

    /**
     * Check that the file size matches the given size.
     * Returns the actual size.
     */
    uint64_t checkFileSizeMatch(const std::string& fileName, const uint64_t& size) const;

    /**
     * Check that the 2 sizes agree
     */
    bool sizeMismatch(const uint64_t& initialSize, const uint64_t& finalSize) const;

    /**
     * Changes the file permissions to read-only
     */
    void makeFileReadOnly(const std::string& fileName) const;

    /**
     * Compare Adler32 checksum of file with calculated one
     */
    void checkAdler32(const std::string& fileName) const;
    
    /**
     * Rename the file
     */
    void renameFile(const std::string& openFileName, const std::string& closedFileName) const;
    
    /**
     * Check if all directories needed for the file output are available.
     * Throws a stor::execption::NoSuchDirectory when a directory does not exist.
     */
    void checkDirectories() const;

    /**
     * Return the relative difference btw to file sizes
     */
    double calcPctDiff(const uint64_t&, const uint64_t&) const;
    

  private:

    //Prevent copying of the FileHandler
    FileHandler(FileHandler const&);
    FileHandler& operator=(FileHandler const&);


  protected:

    FilesMonitorCollection::FileRecordPtr fileRecord_;
    const DbFileHandlerPtr dbFileHandler_;

    utils::TimePoint_t firstEntry_;                // time when first event was writen
    utils::TimePoint_t lastEntry_;                 // time when latest event was writen

    const DiskWritingParams& diskWritingParams_;
    
  private:
    
    const uint64_t maxFileSize_;                    // maximal file size in bytes
    
    const std::string logPath_;                     // log path
    const std::string logFile_;                     // log file including path
    std::string cmsver_;                            // CMSSW version string
  };
  
} // stor namespace

#endif // EventFilter_StorageManager_FileHandler_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
