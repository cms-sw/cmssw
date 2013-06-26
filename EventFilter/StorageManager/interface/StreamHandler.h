// $Id: StreamHandler.h,v 1.16 2012/04/04 12:16:58 mommsen Exp $
/// @file: StreamHandler.h 

#ifndef EventFilter_StorageManager_StreamHandler_h
#define EventFilter_StorageManager_StreamHandler_h

#include <stdint.h>

#include <boost/shared_ptr.hpp>

#include "EventFilter/StorageManager/interface/FilesMonitorCollection.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"
#include "EventFilter/StorageManager/interface/StreamsMonitorCollection.h"


namespace stor {

  class FileHandler;
  class I2OChain;


  /**
   * Abstract class to handle one stream written to disk.
   *
   * $Author: mommsen $
   * $Revision: 1.16 $
   * $Date: 2012/04/04 12:16:58 $
   */
  
  class StreamHandler
  {
  public:
    
    StreamHandler(const SharedResourcesPtr, const DbFileHandlerPtr);

    virtual ~StreamHandler() {};


    /**
     * Gracefully close all open files
     */    
    void closeAllFiles();

    /**
     * Close all files which are have not seen any recent events
     */    
    void closeTimedOutFiles(utils::TimePoint_t currentTime =
                            utils::getCurrentTime());

    /**
     * Close all files which belong to the given lumi section
     * and print number of files for this lumi section into
     * the passed string.
     * Returns true if at least one file was closed.
     */    
    bool closeFilesForLumiSection(const uint32_t& lumiSection, std::string&);

    /**
     * Write the event to the stream file
     */    
    void writeEvent(const I2OChain& event);


  protected:

    typedef boost::shared_ptr<FileHandler> FileHandlerPtr;

    /**
     * Return the stream label
     */
    virtual std::string streamLabel() const = 0;

    /**
     * Return the fraction-to-disk parameter
     */
    virtual double fractionToDisk() const = 0;

    /**
     * Return a new file handler for the provided event
     */    
    virtual FileHandlerPtr newFileHandler(const I2OChain& event) = 0;

    /**
     * Return a new file record for the event
     */    
    FilesMonitorCollection::FileRecordPtr getNewFileRecord(const I2OChain& event);

    /**
     * Return the maximum file size for the stream in MB
     */
    virtual int getStreamMaxFileSize() const = 0;

    /**
     * Return the maximum file size in bytes
     */
    unsigned long long getMaxFileSize() const;


    /**
     * Get the file handler responsible for the event
     */    
    virtual FileHandlerPtr getFileHandler(const I2OChain& event);

  private:

    /**
     * Return true if the file would become too large when
     * adding dataSize in Bytes
     */    
    bool fileTooLarge(const FileHandlerPtr, const unsigned long& dataSize) const;

    /**
     * Get path w/o working directory
     */    
    std::string getBaseFilePath(const uint32_t& runNumber, uint32_t fileCount) const;

    /**
     * Get file system string
     */    
    std::string getFileSystem(const uint32_t& runNumber, uint32_t fileCount) const;

    /**
     * Get the core file name
     */    
    std::string getCoreFileName(const uint32_t& runNumber, const uint32_t& lumiSection) const;
    
    /**
     * Get the instance count of this core file name
     */    
    unsigned int getFileCounter(const std::string& coreFileName);


  protected:

    const SharedResourcesPtr sharedResources_;
    const StatisticsReporterPtr statReporter_;
    const StreamsMonitorCollection::StreamRecordPtr streamRecord_;
    const DiskWritingParams& diskWritingParams_;
    const DbFileHandlerPtr dbFileHandler_;

    typedef std::vector<FileHandlerPtr> FileHandlers;
    FileHandlers fileHandlers_;

    typedef std::map<std::string, unsigned int> CoreFileNamesMap;
    CoreFileNamesMap usedCoreFileNames_;
    

  private:

    //Prevent copying of the StreamHandler
    StreamHandler(StreamHandler const&);
    StreamHandler& operator=(StreamHandler const&);

  };
  
} // namespace stor

#endif // EventFilter_StorageManager_StreamHandler_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
