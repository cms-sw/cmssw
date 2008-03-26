#ifndef FILERECORD_H
#define FILERECORD_H

// $Id: FileRecord.h,v 1.3 2008/02/21 14:16:14 biery Exp $
#include <EventFilter/StorageManager/interface/Parameter.h>
#include <boost/shared_ptr.hpp>
#include <string>

namespace edm {


  class FileRecord
    {
    public:
      FileRecord(int, std::string, std::string);
      ~FileRecord() {}

      void   report(std::ostream &os, int indentation) const;
      void   setFileCounter(int i)  { fileCounter_ = i; }
      void   fileSystem(int);
      void   writeToSummaryCatalog();
      void   writeToMailBox();
      void   notifyTier0();
      void   updateDatabase();
      void   insertFileInDatabase();
      void   moveFileToClosed();
      void   firstEntry(double d)   { firstEntry_ = d; }
      void   lastEntry(double d)    { lastEntry_  = d; }
      void   increaseFileSize(int i){ fileSize_   += (long long) i; }
      void   increaseEventCount()   { events_++; }
      void   checkDirectories();
      void   setCatalog(std::string s)     { statFileName_ = s; }
      void   setRunNumber(int i)           { runNumber_ = i; }
      void   setStreamLabel(std::string s) { streamLabel_ = s;}
      void   setSetupLabel(std::string s)  { setupLabel_ = s;}

      std::string fileName()             { return fileName_; }
      std::string basePath()             { return basePath_; }
      std::string fileSystem()           { return basePath_ + fileSystem_; }
      std::string workingDir()           { return basePath_ + fileSystem_ + workingDir_; }

      std::string fileCounterStr();
      std::string filePath();
      std::string completeFileName();
      std::string timeStamp(double);
     
      int    lumiSection()          { return lumiSection_; }
      int    fileCounter()          { return fileCounter_; }
      long long fileSize()          { return fileSize_; }
      int    events()               { return events_; }

      double lastEntry()            { return lastEntry_; }
      double firstEntry()           { return firstEntry_; }

    private:
      std::string fileName_;                              // file name ( w/o ending )
      std::string basePath_;                              // base path name
      std::string fileSystem_;                            // file system directory
      std::string workingDir_;                            // current working directory

      std::string statFileName_;                          // catalog file name
      std::string mailBoxPath_;                           // mail box path

      std::string setupLabel_;
      std::string streamLabel_;
      
      int    lumiSection_;                           // luminosity section  
      int    runNumber_;                             // runNumber
      int    fileCounter_;                           // number of files with fileName_ as name
      long long fileSize_;                           // current file size
      int    events_;                                // total number of events
      double firstEntry_;                            // time when last event was writen
      double lastEntry_;                             // time when last event was writen
 
      boost::shared_ptr<stor::Parameter> smParameter_;     // 
      
      void   checkDirectory(std::string);
      double calcPctDiff(long long, long long);
   };

 
} // edm namespace
#endif
