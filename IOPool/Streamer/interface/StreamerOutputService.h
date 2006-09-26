#ifndef _StreamerOutputService_h
#define _StreamerOutputService_h 

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSelector.h"
#include "IOPool/Streamer/src/StreamerFileWriter.h"
#include "IOPool/Streamer/interface/InitMessage.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMessage.h"

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

    void init(std::string fileName, unsigned long maxFileSize, double highWaterMark,
              std::string path, std::string mpath, InitMsgView const& init_message) ;

    //By defaulting hlt_trig_count, i don't need to provide any value
    // for hlt_trig_count, which is actually NO MORE used,
    //I will actualy remove this parameter soon, keeping it
    // ONLY for backward compatability
    // AA - 09/22/2006 
    void writeEvent(EventMsgView const& msg, uint32 hlt_trig_count=0);
    
    void stop(); // shouldn't be called from destructor.

    std::list<std::string>& get_filelist() { return files_; }
    std::string& get_currfile() { return fileName_;}

  private:
    void writeHeader(InitMsgView const& init_message);
    bool wantsEvent(EventMsgView const& eventView); 
    void initializeSelection(InitMsgView const& initView);
 
     unsigned long maxFileSize_;
     unsigned long maxFileEventCount_;
     unsigned long currentFileSize_;
     unsigned long totalEventCount_;
     unsigned long eventsInFile_;
     unsigned long fileNameCounter_;

     void checkFileSystem();
     void writeToMailBox();
     std::list<std::string> files_;

     std::string filen_;
     double highWaterMark_;
     std::string path_;
     std::string mpath_;
     double diskUsage_;
     std::string closedFiles_;

     // memory to keep the INIT message for when writing to more than one file
     char saved_initmsg_[1000*1000*2];

     std::string fileName_;
     std::string indexFileName_;

     boost::shared_ptr<StreamerFileWriter> streamNindex_writer_;
     
     //ParameterSet that contains SelectEvents criteria
     edm::ParameterSet requestParamSet_; 

     // event selector that does the work of accepting/rejecting events
     boost::shared_ptr<edm::EventSelector> eventSelector_;
  };
}
#endif

