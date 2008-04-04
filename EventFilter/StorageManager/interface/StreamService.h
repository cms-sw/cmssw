#ifndef STREAMSERVICE_H
#define STREAMSERVICE_H

// $Id: StreamService.h,v 1.6 2008/03/10 10:05:57 meschi Exp $

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

  typedef std::vector <boost::shared_ptr<FileRecord> >                                          OutputSummary;
  typedef std::vector <boost::shared_ptr<FileRecord> >::iterator                                OutputSummaryIterator;
  typedef std::vector <boost::shared_ptr<FileRecord> >::reverse_iterator                        OutputSummaryReIterator;
  typedef std::map <boost::shared_ptr<FileRecord>, boost::shared_ptr<OutputService> >           OutputMap;
  typedef std::map <boost::shared_ptr<FileRecord>, boost::shared_ptr<OutputService> >::iterator OutputMapIterator;

  class StreamService
    {
    public:
      StreamService(ParameterSet const&, InitMsgView const&);
      ~StreamService() {}
      
      bool   nextEvent(EventMsgView const&);
      void   stop();
      void   report(std::ostream &os, int indentation) const;

      void   setNumberOfFileSystems(int i)   { numberOfFileSystems_ = i; } 
      void   setCatalog(std::string s)       { catalog_  = s; }
      void   setSourceId(std::string s)      { sourceId_ = s; }
      void   setFileName(std::string s)      { fileName_ = s; }
      void   setFilePath(std::string s)      { filePath_ = s; }
      void   setMaxFileSize(int x); 
      void   setMathBoxPath(std::string s)   { mailboxPath_ = s; }
      void   setSetupLabel(std::string s)    { setupLabel_ = s; }
      void   setHighWaterMark(double d)      { highWaterMark_ = d; }
      void   setLumiSectionTimeOut(double d) { lumiSectionTimeOut_ = d; }

      std::list<std::string> getFileList();
      std::list<std::string> getCurrentFileList();

    private:
      boost::shared_ptr<OutputService>  newOutputService();
      boost::shared_ptr<OutputService>  getOutputService(EventMsgView const&);
      boost::shared_ptr<FileRecord>     generateFileRecord();  

      void   saveInitMessage(InitMsgView const&);
      void   initializeSelection(InitMsgView const&);
      bool   acceptEvent(EventMsgView const&);
      void   setStreamParameter();
      void   closeTimedOutFiles();
      double getCurrentTime();
      bool   checkEvent(boost::shared_ptr<FileRecord>, EventMsgView const&);
      bool   checkFileSystem();
      void   handleLock(boost::shared_ptr<FileRecord>);
      
      //
      ParameterSet                           parameterSet_;
      boost::shared_ptr<edm::EventSelector>  eventSelector_;
      OutputMap                              outputMap_;
      OutputSummary                          outputSummary_;
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
      std::string fileName_;
      std::string filePath_;
      int    maxFileSizeInMB_;
      std::string mailboxPath_;
      std::string setupLabel_;
      std::string streamLabel_;
      long long maxSize_;
      double highWaterMark_;
      double lumiSectionTimeOut_;

      //@@EM added lock to handle access to file list by monitoring loop
      boost::mutex list_lock_;

     };

} // edm namespace
#endif
