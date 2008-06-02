#ifndef FILERECORD_H
#define FILERECORD_H

// $Id: FileRecord.h,v 1.6 2008/05/13 18:06:46 loizides Exp $
#include <EventFilter/StorageManager/interface/Parameter.h>
#include <boost/shared_ptr.hpp>
#include <string>

namespace edm {


  class FileRecord
    {
    public:
      FileRecord(int lumi, const std::string &file, const std::string &path);
      ~FileRecord() {}

      void   report(std::ostream &os, int indentation) const;
      void   setFileCounter(int i)         { fileCounter_ = i; }
      void   fileSystem(int);
      void   writeToSummaryCatalog();
      void   writeToMailBox();
      void   notifyTier0();
      void   updateDatabase();
      void   insertFileInDatabase();
      void   moveFileToClosed();
      void   firstEntry(double d)          { firstEntry_ = d; }
      void   lastEntry(double d)           { lastEntry_  = d; }
      void   increaseFileSize(int i)       { fileSize_   += (long long) i; }
      void   increaseEventCount()          { events_++; }
      void   checkDirectories()      const;
      void   setRunNumber(int i)                     { runNumber_ = i; }
      void   setStreamLabel(const std::string &s)    { streamLabel_ = s;}
      void   setSetupLabel(const std::string &s)     { setupLabel_ = s;}

      const std::string& fileName()  const { return fileName_; }
      const std::string& basePath()  const { return basePath_; }
      std::string fileSystem()       const { return basePath_ + fileSystem_; }
      std::string workingDir()       const { return basePath_ + fileSystem_ + workingDir_; }
      std::string fileCounterStr()   const;
      std::string filePath()         const;
      std::string completeFileName() const;
      std::string timeStamp(double)  const;
      std::string logFile()          const;
      std::string notFile()          const;
     
      int         lumiSection()      const { return lumiSection_; }
      int         fileCounter()      const { return fileCounter_; }
      long long   fileSize()         const { return fileSize_; }
      int         events()           const { return events_; } 
      double      lastEntry()        const { return lastEntry_; }
      double      firstEntry()       const { return firstEntry_; }

    private:
      boost::shared_ptr<stor::Parameter> smParameter_; 

      std::string fileName_;                         // file name (w/o ending)
      std::string basePath_;                         // base path name
      std::string fileSystem_;                       // file system directory
      std::string workingDir_;                       // current working directory
      std::string mailBoxPath_;                      // mail box path
      std::string logPath_;                          // log path
      std::string logFile_;                          // log file including path
      std::string setupLabel_;                       // setup label
      std::string streamLabel_;                      // datastream label
      std::string cmsver_;                           // CMSSW version string
      
      int         lumiSection_;                      // luminosity section  
      int         runNumber_;                        // runNumber
      int         fileCounter_;                      // number of files with fileName_ as name
      long long   fileSize_;                         // current file size
      int         events_;                           // total number of events
      double      firstEntry_;                       // time when last event was writen
      double      lastEntry_;                        // time when last event was writen
      
      void   checkDirectory(const std::string &) const;
      double calcPctDiff(long long, long long) const;
   };
 
} // edm namespace
#endif
