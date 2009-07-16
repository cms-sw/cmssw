// $Id: StreamHandler.h,v 1.2 2009/06/10 08:15:24 dshpakov Exp $

#ifndef StorageManager_StreamHandler_h
#define StorageManager_StreamHandler_h

#include <boost/shared_ptr.hpp>

#include "EventFilter/StorageManager/interface/FileHandler.h"
#include "EventFilter/StorageManager/interface/I2OChain.h"
#include "EventFilter/StorageManager/interface/SharedResources.h"
#include "EventFilter/StorageManager/interface/StreamsMonitorCollection.h"
#include "EventFilter/StorageManager/interface/StatisticsReporter.h"


namespace stor {

  /**
   * Abstract class to handle one stream written to disk.
   *
   * $Author: dshpakov $
   * $Revision: 1.2 $
   * $Date: 2009/06/10 08:15:24 $
   */
  
  class StreamHandler
  {
  public:
    
    explicit StreamHandler(SharedResourcesPtr);

    virtual ~StreamHandler();


    /**
     * Gracefully close all open files
     */    
    void closeAllFiles();

    /**
     * Close all files which are have not seen any recent events
     */    
    void closeTimedOutFiles(utils::time_point_t currentTime =
                            utils::getCurrentTime());

    /**
     * Write the event to the stream file
     */    
    void writeEvent(const I2OChain& event);


  protected:

    typedef boost::shared_ptr<FileHandler> FileHandlerPtr;

    /**
     * Return the stream label
     */
    virtual const std::string streamLabel() const = 0;

    /**
     * Return a new file handler for the provided event
     */    
    virtual const FileHandlerPtr newFileHandler(const I2OChain& event) = 0;

    /**
     * Return a new file record for the event
     */    
    const FilesMonitorCollection::FileRecordPtr getNewFileRecord(const I2OChain& event);

    /**
     * Return the maximum file size for the stream in MB
     */
    virtual const int getStreamMaxFileSize() const = 0;

    /**
     * Return the maximum file size in bytes
     */
    const long long getMaxFileSize() const;


  private:

    /**
     * Get the file handler responsible for the event
     */    
    const FileHandlerPtr getFileHandler(const I2OChain& event);

    /**
     * Return true if the file would become too large when
     * adding dataSize in Bytes
     */    
    const bool fileTooLarge(const FileHandlerPtr, const unsigned long& dataSize) const;

    /**
     * Get path w/o working directory
     */    
    const std::string getBaseFilePath(const uint32& runNumber, uint32_t fileCount) const;

    /**
     * Get file system string
     */    
    const std::string getFileSystem(const uint32& runNumber, uint32_t fileCount) const;

    /**
     * Get the core file name
     */    
    const std::string getCoreFileName(const uint32& runNumber, const uint32& lumiSection) const;
    
    /**
     * Get the instance count of this core file name
     */    
    const unsigned int getFileCounter(const std::string& coreFileName);


  protected:

    const StatisticsReporterPtr _statReporter;
    const StreamsMonitorCollection::StreamRecordPtr _streamRecord;
    const DiskWritingParams _diskWritingParams;

    typedef std::vector<FileHandlerPtr> FileHandlers;
    FileHandlers _fileHandlers;

    typedef std::map<std::string, unsigned int> CoreFileNamesMap;
    CoreFileNamesMap _usedCoreFileNames;
    

  private:

    //Prevent copying of the StreamHandler
    StreamHandler(StreamHandler const&);
    StreamHandler& operator=(StreamHandler const&);

  };
  
} // namespace stor

#endif // StorageManager_StreamHandler_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
