#ifndef _StreamerOutputService_h
#define _StreamerOutputService_h 

// $Id: StreamerOutputService.h,v 1.10 2006/12/19 00:30:44 wmtan Exp $

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSelector.h"
#include "IOPool/Streamer/src/StreamerFileWriter.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/EventMessage.h"
#include "IOPool/Streamer/interface/StreamerStatService.h"

#include <iostream>
#include <vector>
#include <memory>
#include <string>

#include <boost/shared_ptr.hpp>

namespace edm
{
  class StreamerOutputService 
  {
  public:

    explicit StreamerOutputService(edm::ParameterSet const& ps);
    explicit StreamerOutputService();
    ~StreamerOutputService();

    void init(std::string fileName, unsigned int maxFileSize, double highWaterMark,
              std::string path, std::string mpath, 
	      std::string catalog, uint32 disks, 
	      InitMsgView const& init_message) ;

    //By defaulting hlt_trig_count, i don't need to provide any value
    // for hlt_trig_count, which is actually NO MORE used,
    //I will actualy remove this parameter soon, keeping it
    // ONLY for backward compatability
    // AA - 09/22/2006 
 
    bool writeEvent(EventMsgView const& msg, uint32 hlt_trig_count=0);
    void closeFile(EventMsgView const& msg); 
    
    void stop(); // shouldn't be called from destructor.

    std::list<std::string>& get_filelist() { return files_; }
    std::string& get_currfile() { return fileName_;}

  private:
    void writeHeader(InitMsgView const& init_message);
    bool wantsEvent(EventMsgView const& eventView); 
    void initializeSelection(InitMsgView const& initView);
 
     unsigned int maxFileSize_;
     unsigned int maxFileEventCount_;
     unsigned int currentFileSize_;
     unsigned int totalEventCount_;
     unsigned int eventsInFile_;
     unsigned int fileNameCounter_;

     void checkFileSystem();
     void writeToMailBox();
     std::list<std::string> files_;

     std::string filen_;
     double highWaterMark_;
     std::string path_;
     std::string mpath_;
     double diskUsage_;
     std::string closedFiles_;
     std::string catalog_;
     uint32 nLogicalDisk_;

     // memory to keep the INIT message for when writing to more than one file
     char saved_initmsg_[1000*1000*2];

     std::string fileName_;
     std::string indexFileName_;
     std::string lockFileName_;

     boost::shared_ptr<StreamerFileWriter> streamNindex_writer_;
     
     //ParameterSet that contains SelectEvents criteria
     edm::ParameterSet requestParamSet_; 

     // event selector that does the work of accepting/rejecting events
     boost::shared_ptr<edm::EventSelector> eventSelector_;

     // statistic writer
     boost::shared_ptr<edm::StreamerStatWriteService> statistics_;

  };
}
#endif

