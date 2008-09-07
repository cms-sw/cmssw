#ifndef STREAMSERVICE_H
#define STREAMSERVICE_H

// $Id: StreamService.h,v 1.11 2008/08/13 22:48:11 biery Exp $

#include <EventFilter/StorageManager/interface/FileRecord.h>
#include <EventFilter/StorageManager/interface/OutputService.h>

#include <IOPool/Streamer/interface/MsgHeader.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>

#include <boost/shared_ptr.hpp>
#include <boost/thread/thread.hpp>

#include <string>
#include <map>


namespace edm {

  typedef std::map <boost::shared_ptr<FileRecord>, boost::shared_ptr<OutputService> >           OutputMap;
  typedef std::map <boost::shared_ptr<FileRecord>, boost::shared_ptr<OutputService> >::iterator OutputMapIterator;

  class StreamService
  {
    public:
      virtual ~StreamService();
      
      virtual bool nextEvent(const uint8 * const) = 0;
      virtual void stop() = 0;
      virtual void report(std::ostream &os, int indentation) const = 0;
      int    lumiSection() const { return lumiSection_; }

      void   setNumberOfFileSystems(int i)          { numberOfFileSystems_ = i; } 
      void   setCatalog(const std::string &s)       { catalog_  = s; }
      void   setSourceId(const std::string &s)      { sourceId_ = s; }
      void   setFileName(const std::string &s)      { fileName_ = s; }
      void   setFilePath(const std::string &s)      { filePath_ = s; }
      void   setMaxFileSize(int x); 
      void   setSetupLabel(std::string s)           { setupLabel_ = s; }
      void   setHighWaterMark(double d)             { highWaterMark_ = d; }
      void   setLumiSectionTimeOut(double d)        { lumiSectionTimeOut_ = d; }
      virtual void closeTimedOutFiles(int lumi, double timeoutstart) = 0;
      virtual void closeTimedOutFiles() = 0;
 
      double getCurrentTime() const;

      std::list<std::string> getFileList();
      std::list<std::string> getCurrentFileList();
      const std::string& getStreamLabel() const { return streamLabel_; }

    protected:
      void   setStreamParameter();
      bool   checkFileSystem() const;
      void   fillOutputSummaryClosed(const boost::shared_ptr<FileRecord> &file);

      // variables
      ParameterSet                           parameterSet_;
      OutputMap                              outputMap_;
      std::map<std::string, int>             outputSummary_;
      std::list<std::string>                 outputSummaryClosed_;

      // set from event message
      int    runNumber_;
      int    lumiSection_;

      // should be output module parameter
      int    numberOfFileSystems_;
      std::string catalog_;
      std::string sourceId_;

      // output module parameter
      std::string fileName_;
      std::string filePath_;
      int    maxFileSizeInMB_;
      std::string setupLabel_;
      std::string streamLabel_;
      long long maxSize_;
      double highWaterMark_;
      double lumiSectionTimeOut_;

      int ntotal_; //total number of files

      //@@EM added lock to handle access to file list by monitoring loop
      boost::mutex list_lock_;

  };

} // edm namespace
#endif
