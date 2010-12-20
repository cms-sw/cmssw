// $Id: FileHandler.h,v 1.12.2.1 2010/11/04 14:46:00 mommsen Exp $
/// @file: FileHandler.h 

#ifndef StorageManager_FileHandler_h
#define StorageManager_FileHandler_h

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
   * $Revision: 1.12.2.1 $
   * $Date: 2010/11/04 14:46:00 $
   */

  class FileHandler
  {
  public:
        
    FileHandler
    (
      FilesMonitorCollection::FileRecordPtr,
      const DbFileHandlerPtr,
      const DiskWritingParams&,
      const unsigned long long& maxFileSize
     );
    
    virtual ~FileHandler() {};
 
    /**
     * Write the event contained in the I2OChain
     */
    void writeEvent(const I2OChain&);

    /**
     * Returns true if the file has not seen any recent events
     */
    bool tooOld(const utils::time_point_t currentTime = utils::getCurrentTime());

    /**
     * Returns true if the file corresponds to the given lumi section
     */
    bool isFromLumiSection(const uint32_t lumiSection);

    /**
     * Returns true if the additional data size would push the file size
     * beyond maxFileSize.
     */
    bool tooLarge(const unsigned long& dataSize);


    ////////////////////////////
    // File parameter getters //
    ////////////////////////////
    
    /**
     * Return the number of events in the file
     */
    int events() const;
    
    /**
     * Return the luminosity section the file belongs to
     */
    uint32_t lumiSection() const
    { return _fileRecord->lumiSection; }
    
    /**
     * Return the size of the file in bytes
     */
    unsigned long long fileSize() const;

    /**
     * Close the file
     */
    virtual void closeFile(const FilesMonitorCollection::FileRecord::ClosingReason&) = 0;


  protected:

    /**
     * Write the I2OChain to the file
     */
    virtual void do_writeEvent(const I2OChain& event) = 0;

    
    ////////////////////////////
    // File parameter setters //
    ////////////////////////////

    /**
     * Set the adler checksum for the file
     */
    void setAdler(uint32_t s, uint32_t i)
    { _adlerstream = s; _adlerindex = i; }
    
    
    //////////////////////
    // File bookkeeping //
    //////////////////////
    
    /**
     * Write summary information in file catalog
     */
    void writeToSummaryCatalog() const;

    /**
     * Write command to update the file information in the CMS_STOMGR.TIER0_INJECTION table
     * into the _logFile.
     */
    void updateDatabase() const;


    /**
     * Write command to insert a new file into the CMS_STOMGR.TIER0_INJECTION table
     * into the _logFile.
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
      const bool& useIndexFile,
      const FilesMonitorCollection::FileRecord::ClosingReason&
    );


  private:

    /**
     * Check that the file size matches the given size.
     * Returns the actual size.
     */
    unsigned long long checkFileSizeMatch(const std::string& fileName, const unsigned long long& size) const;

    /**
     * Check that the 2 sizes agree
     */
    bool sizeMismatch(const unsigned long long& initialSize, const unsigned long long& finalSize) const;

    /**
     * Changes the file permissions to read-only
     */
    void makeFileReadOnly(const std::string& fileName) const;

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
     * Return the name of the log file
     */
    std::string logFile(const DiskWritingParams&) const;

    /**
     * Return the relative difference btw to file sizes
     */
    double calcPctDiff(const unsigned long long&, const unsigned long long&) const;
    

  private:

    //Prevent copying of the FileHandler
    FileHandler(FileHandler const&);
    FileHandler& operator=(FileHandler const&);


  protected:

    FilesMonitorCollection::FileRecordPtr _fileRecord;
    const DbFileHandlerPtr _dbFileHandler;

    utils::time_point_t _firstEntry;                // time when first event was writen
    utils::time_point_t _lastEntry;                 // time when latest event was writen

    const DiskWritingParams& _diskWritingParams;
    
  private:
    
    const unsigned long long _maxFileSize;          // maximal file size in bytes
    
    const std::string  _logPath;                    // log path
    const std::string  _logFile;                    // log file including path
    std::string  _cmsver;                           // CMSSW version string

    uint32_t _adlerstream;                          // adler32 checksum for streamer file
    uint32_t _adlerindex;                           // adler32 checksum for index file
  };
  
} // stor namespace

#endif // StorageManager_FileHandler_h


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
