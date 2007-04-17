#ifndef STREAMSERVICE_H
#define STREAMSERVICE_H

// $Id:$

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
#include "IOPool/Streamer/interface/FileRecord.h"
#include "IOPool/Streamer/interface/OutputService.h"  

#include <boost/shared_ptr.hpp>
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

      void   setNumberOfFileSystems(int i) { numberOfFileSystems_ = i; } 
      void   setCatalog(string s) { catalog_ = s; }

      std::list<std::string> getFileList();
      std::list<std::string>getCurrentFileList();

    private:
      boost::shared_ptr<OutputService>  newOutputService();
      boost::shared_ptr<OutputService>  getOutputService(EventMsgView const&);
      boost::shared_ptr<FileRecord> generateFileRecord();  

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
      string                                 currentLockPath_;

      // set from event message
      int    runNumber_;
      int    lumiSection_;

      // set from init message ( is init message )
      char   saved_initmsg_[1000*1000*2];
      
      // should be output module parameter
      int    numberOfFileSystems_;
      string catalog_;

      // output module parameter
      string fileName_;
      string filePath_;
      string mailboxPath_;
      string setupLabel_;
      string streamLabel_;
      int    maxSize_;
      double highWaterMark_;
      double lumiSectionTimeOut_;
     };

} // edm namespace
#endif
