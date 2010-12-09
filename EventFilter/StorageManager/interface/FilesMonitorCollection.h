// $Id: FilesMonitorCollection.h,v 1.12 2010/01/29 15:47:59 mommsen Exp $
/// @file: FilesMonitorCollection.h 

#ifndef StorageManager_FilesMonitorCollection_h
#define StorageManager_FilesMonitorCollection_h

#include <iomanip>
#include <sstream>
#include <stdint.h>
#include <vector>

#include <boost/circular_buffer.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/shared_ptr.hpp>

#include "xdata/UnsignedInteger32.h"

#include "EventFilter/StorageManager/interface/MonitorCollection.h"


namespace stor {

  /**
   * A collection of monitoring entities for open and closed files
   *
   * $Author: mommsen $
   * $Revision: 1.12 $
   * $Date: 2010/01/29 15:47:59 $
   */
  
  class FilesMonitorCollection : public MonitorCollection
  {
  public:

    struct FileRecord
    {
      enum ClosingReason
      {
        notClosed = 0,
        runEnded,
        LSended,
        timeout,
        size,
        truncated,
        inaccessible
      };

      enum FileStatus
      {
        open,
        closed,
        current
      };

      uint32_t           entryCounter;      // file counter
      uint32_t           runNumber;         // run number
      uint32_t           lumiSection;       // luminosity section 
      std::string        streamLabel;       // datastream label
      std::string        baseFilePath;      // file path w/o the working directory
      std::string        coreFileName;      // file name w/o instance & file ending
      unsigned int       fileCounter;       // counter of number of coreFileNames used
      ClosingReason      whyClosed;         // reason why the given file was closed
      bool               isOpen;            // true if file is in open directory
      unsigned long long fileSize;          // file size in bytes
      uint32_t           eventCount;        // number of events
      std::string closingReason();          // reason why file was closed
      std::string fileName();               // full file name w/o file ending
      std::string filePath(FileStatus status=current); // complete file path for the given file status
      std::string completeFileName(FileStatus status=current)
      { return ( filePath(status) + "/" + fileName() ); }

    };

    // We do not know how many files there will be.
    // Thus, we need a vector of them.
    typedef boost::shared_ptr<FileRecord> FileRecordPtr;
    typedef boost::circular_buffer<FileRecordPtr> FileRecordList;


    explicit FilesMonitorCollection(const utils::duration_t& updateInterval);

    const FileRecordPtr getNewFileRecord();

    void getFileRecords(FileRecordList&) const;


  private:

    //Prevent copying of the FilesMonitorCollection
    FilesMonitorCollection(FilesMonitorCollection const&);
    FilesMonitorCollection& operator=(FilesMonitorCollection const&);

    virtual void do_calculateStatistics();
    virtual void do_reset();
    virtual void do_appendInfoSpaceItems(InfoSpaceItems&);
    virtual void do_updateInfoSpaceItems();

    FileRecordList _fileRecords;
    mutable boost::mutex _fileRecordsMutex;

    const unsigned int _maxFileEntries; // maximum number of files to remember
    uint32_t _entryCounter;

    xdata::UnsignedInteger32 _closedFiles;                 // number of closed files
    xdata::UnsignedInteger32 _openFiles;                   // number of open files

  };
  
} // namespace stor

#endif // StorageManager_FilesMonitorCollection_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
