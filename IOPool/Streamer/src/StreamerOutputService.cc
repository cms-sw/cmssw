#include "IOPool/Streamer/interface/EventStreamOutput.h"
#include "IOPool/Streamer/interface/StreamerOutputService.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/EOFRecordBuilder.h"
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/DumpTools.h"

// to stat files and directories
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <string>

using namespace edm;
using namespace std;

namespace edm
{

std::string itoa(int i){
        char temp[20];
        sprintf(temp,"%d",i);
        return((std::string)temp);
}

 //StreamerOutputService::StreamerOutputService(edm::ParameterSet const& ps):
 //maxFileSize_(ps.template getParameter<int>("maxFileSize")),
 //maxFileEventCount_(ps.template getParameter<int>("maxFileEventCount")),
 // defaulting - need numbers from Emilio
 StreamerOutputService::StreamerOutputService():
 maxFileSize_(1073741824),
 maxFileEventCount_(10000),
 currentFileSize_(0),
 totalEventCount_(0),
 eventsInFile_(0),
 fileNameCounter_(0),
 highWaterMark_(0.9), diskUsage_(0.0)
   
  {
    saved_initmsg_[0] = '\0';
    
  }

void StreamerOutputService::init(std::string fileName, unsigned long maxFileSize, double highWaterMark,
                                 std::string path, std::string mpath, InitMsgView& view)
  { 
   maxFileSize_ = maxFileSize;
   highWaterMark_ = highWaterMark;
   path_ = path;
   mpath_ = mpath;
   filen_ = fileName;
   fileName_ = path_ + "/" + filen_ + "." + itoa((int)fileNameCounter_) + ".dat";
   indexFileName_ = path_ + "/" + filen_ + "." + itoa((int)fileNameCounter_) + ".ind";
   
   streamNindex_writer_ = boost::shared_ptr<StreamerFileWriter>(new StreamerFileWriter(fileName_, indexFileName_));

   //dumpInitHeader(&view);
   
   writeHeader(view);
    
   //INIT msg can be saved as INIT msg itself.

   // save the INIT message for when writing to the next file
   // that is openned
   unsigned char* pos = (unsigned char*) &saved_initmsg_[0];
   unsigned char* from = view.startAddress();
   int dsize = (int)view.size();
   copy(from,from+dsize,pos);
  }

StreamerOutputService::~StreamerOutputService()
  {
   // expect to do this manually at the end of a run
   // stop();   //Remove this from destructor if you want higher class to do that at its will.
              // and if stop() is ever made Public.
   writeToMailBox();
   std::ostringstream entry;
   entry << fileNameCounter_ << " "
         << fileName_
         << " " << eventsInFile_
         << "   " << currentFileSize_;
   files_.push_back(entry.str());
   closedFiles_ += ", ";
   closedFiles_ += fileName_;
   // HEREHERE for test
   std::cout << "#    name                             evt        size     " << endl;
   for(std::list<std::string>::const_iterator it = files_.begin(); it != files_.end(); it++)
     std::cout << *it << endl;
   std::cout << "Disk Usage = " << diskUsage_ << endl;
   std::cout << "Closed files = " << closedFiles_ << endl;

  }

void StreamerOutputService::stop()
  {

    //Does EOF record writting and HLT Event count for each path for EOF
    streamNindex_writer_->stop();
    // gives the EOF Size
    currentFileSize_ += streamNindex_writer_->getStreamEOFSize();
    
  }

void StreamerOutputService::writeHeader(InitMsgView& init_message)
  {
    //Write the Init Message to Streamer and Index file
    streamNindex_writer_->doOutputHeader(init_message);

    currentFileSize_ += init_message.size();

  }

void StreamerOutputService::writeEvent(EventMsgView& eview, uint32 hltsize=0)
  { 
        // check for number of events is no longer required 09/22/2006
        //if ( eventsInFile_ > maxFileEventCount_  || currentFileSize_ > maxFileSize_ )
        if ( currentFileSize_ > maxFileSize_ )
           {
             checkFileSystem(); // later should take some action
             stop();
             writeToMailBox();
             fileNameCounter_++;

             string tobeclosedFile = fileName_;

             std::cout<<" better to use temp variable as not sure if writer is still using"<<std::endl;
             // them? shouldn't be! (no time to look now)

             // also should be checking the filesystem here at path_
             fileName_ = path_ + "/" + filen_ + "." + itoa((int)fileNameCounter_) + ".dat";
             indexFileName_ = path_ + "/" + filen_ + "." + itoa((int)fileNameCounter_) + ".ind";

             streamNindex_writer_.reset(new StreamerFileWriter(fileName_, indexFileName_));

             // write out the summary line for this last file
             std::ostringstream entry;
             entry << (fileNameCounter_ - 1) << " " 
                   << fileName_
                   << " " << eventsInFile_
                   << "   " << currentFileSize_;
             files_.push_back(entry.str());
             if(fileNameCounter_!=1) closedFiles_ += ", ";
             closedFiles_ += tobeclosedFile;

             eventsInFile_ = 0; 
             currentFileSize_ = 0;
             // write the Header for the newly openned file
             // from the previously saved INIT msg
             InitMsgView myview(&saved_initmsg_[0]);

             writeHeader(myview);
           }
                     
             
           //Write the Event Message to Streamer and index
           streamNindex_writer_->doOutputEvent(eview);

           eventsInFile_++;
           totalEventCount_++;
           currentFileSize_ += eview.size();
  }

  void StreamerOutputService::writeToMailBox()
  {
    std::ostringstream ofilename;
    ofilename << mpath_ << "/" << filen_ << "." << fileNameCounter_ << ".smry";
    ofstream of(ofilename.str().c_str());
    of << fileName_;
    of.close();
  }
  
  void StreamerOutputService::checkFileSystem()
  {
    struct statfs64 buf;
    int retVal = statfs64(path_.c_str(), &buf);
    if(retVal!=0)
      //edm::LogWarning("StreamerOutputService") << "Could not stat output filesystem for path " 
      std::cout << "StreamerOutputService: " << "Could not stat output filesystem for path " 
                                               << path_ << std::endl;
  
    unsigned long btotal = 0;
    unsigned long bfree = 0;
    unsigned long blksize = 0;
    if(retVal==0)
      {
        blksize = buf.f_bsize;
        btotal = buf.f_blocks;
        bfree  = buf.f_bfree;
      }
    float dfree = float(bfree)/float(btotal);
    float dusage = 1. - dfree;
    diskUsage_ = dusage;
    if(dusage>highWaterMark_)
      //edm::LogWarning("StreamerOutputService") << "Output filesystem for path " << path_ 
      std::cout << "StreamerOutputService: " << "Output filesystem for path " << path_ 
                                 << " is more than " << highWaterMark_*100 << "% full " << std::endl;

  }

}  //end-of-namespace block
