#ifndef STREAMSERVICE_H
#define STREAMSERVICE_H

// $Id: StreamService.h,v 1.9 2008/05/13 18:06:46 loizides Exp $

// - handling output files per stream make the problem 1-dimensional 
// - allows to use different file handling rules per stream
  
// functionality:
// - create and delete output service
// - pass init and event message to correct output service
// - do accounting
// - enforce file management rules

// needs:
// - event selector
// - copy of init message to create a new file
// - filename, rules, etc.

#include "FWCore/Framework/interface/EventSelector.h"

#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"

#include "EventFilter/StorageManager/interface/FileRecord.h"
#include "EventFilter/StorageManager/interface/OutputService.h"  

#include <boost/shared_ptr.hpp>
#include "boost/thread/thread.hpp"

#include <string>
#include <map>

namespace edm {

  typedef std::map <boost::shared_ptr<FileRecord>, boost::shared_ptr<OutputService> >           OutputMap;
  typedef std::map <boost::shared_ptr<FileRecord>, boost::shared_ptr<OutputService> >::iterator OutputMapIterator;

  class StreamService
    {
    public:
      StreamService(ParameterSet const&, InitMsgView const&);
      ~StreamService() { stop(); }
      
      bool   nextEvent(EventMsgView const&);
      void   stop();
      void   report(std::ostream &os, int indentation) const;

      void   setNumberOfFileSystems(int i)          { numberOfFileSystems_ = i; } 
      void   setCatalog(const std::string &s)       { catalog_  = s; }
      void   setSourceId(const std::string &s)      { sourceId_ = s; }
      void   setFileName(const std::string &s)      { fileName_ = s; }
      void   setFilePath(const std::string &s)      { filePath_ = s; }
      void   setMaxFileSize(int x); 
      void   setMathBoxPath(std::string s)          { mailboxPath_ = s; }
      void   setSetupLabel(std::string s)           { setupLabel_ = s; }
      void   setHighWaterMark(double d)             { highWaterMark_ = d; }
      void   setLumiSectionTimeOut(double d)        { lumiSectionTimeOut_ = d; }

      std::list<std::string> getFileList();
      std::list<std::string> getCurrentFileList();
      const std::string& getStreamLabel()    const {return streamLabel_;}

    private:
      boost::shared_ptr<OutputService>  newOutputService();
      boost::shared_ptr<OutputService>  getOutputService(EventMsgView const&);
      boost::shared_ptr<FileRecord>     generateFileRecord();  

      void        saveInitMessage(InitMsgView const&);
      void        initializeSelection(InitMsgView const&);
      bool        acceptEvent(EventMsgView const&);
      void        setStreamParameter();
      void        closeTimedOutFiles();
      double      getCurrentTime() const;
      bool        checkEvent(boost::shared_ptr<FileRecord>, EventMsgView const&) const;
      bool        checkFileSystem() const;
      void        fillOutputSummaryClosed(const boost::shared_ptr<FileRecord> &file);

      // variables
      ParameterSet                           parameterSet_;
      boost::shared_ptr<edm::EventSelector>  eventSelector_;
      OutputMap                              outputMap_;
      std::map<std::string, int>             outputSummary_;
      std::list<std::string>                 outputSummaryClosed_;
      std::string                            currentLockPath_;

      // set from event message
      int    runNumber_;
      int    lumiSection_;

      // set from init message ( is init message )
      char   saved_initmsg_[1000*1000*2];
      
      // should be output module parameter
      int    numberOfFileSystems_;
      std::string catalog_;
      std::string sourceId_;

      // output module parameter
      std::string mailboxPath_;
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
