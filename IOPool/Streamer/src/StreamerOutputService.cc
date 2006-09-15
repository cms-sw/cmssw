#include "IOPool/Streamer/interface/EventStreamOutput.h"
#include "IOPool/Streamer/interface/StreamerOutputService.h"
#include "IOPool/Streamer/interface/InitMsgBuilder.h"
#include "IOPool/Streamer/interface/EventMsgBuilder.h"
#include "IOPool/Streamer/interface/EOFRecordBuilder.h"
#include "IOPool/Streamer/interface/MsgTools.h"
#include "IOPool/Streamer/interface/DumpTools.h"


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
   stream_writer_ = std::auto_ptr<StreamerOutputFile>(new StreamerOutputFile(fileName_));
   index_writer_ =  std::auto_ptr<StreamerOutputIndexFile>(new StreamerOutputIndexFile(indexFileName_));

   // rebuild (should not be doing this!!)
   //dumpInitHeader(&view);
   
   uint8 psetid2[16];
   Strings hlt2;
   Strings l12;
   view.pset(psetid2);
   view.hltTriggerNames(hlt2);
   view.l1TriggerNames(l12);
   uint32 desclength = view.descLength();
   InitMsgBuilder init_message(view.startAddress(),view.size(),
                       view.run(),
                       Version(view.protocolVersion(), psetid2),
                       view.releaseTag().c_str(),
                       hlt2,l12);

   init_message.setDescLength(desclength);
   writeHeader(init_message);

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
    //Write the EOF Record Both at the end of Streamer file and Index file
    uint32 dummyStatusCode = 1234;
    std::vector<uint32> hltStats;

    hltStats.push_back(32);
    hltStats.push_back(33);
    hltStats.push_back(34);

    // is this writing the number of events written?
    stream_writer_->writeEOF(dummyStatusCode, hltStats);
    index_writer_->writeEOF(dummyStatusCode, hltStats);
    // how am I supposed to get the size?
   // build using dummy values
    EOFRecordBuilder eof(1, 
                         eventsInFile_,
                         dummyStatusCode,
                         hltStats,
                         0,
                         1);
    currentFileSize_ += eof.size();
  }

void StreamerOutputService::writeHeader(InitMsgBuilder& init_message)
  {
    //Write the Init Message to Streamer file
    stream_writer_->write(init_message); 

    // Don't know what should go in as Magic and Reserved fields.
    uint32 magic = 22;
    uint64 reserved = 666;
    index_writer_->writeIndexFileHeader(magic, reserved);

    index_writer_->write(init_message);
    
    currentFileSize_ += init_message.size();

  }

void StreamerOutputService::writeEvent(EventMsgView& eview, uint32 hltsize)
  {
        if ( eventsInFile_ > maxFileEventCount_  || currentFileSize_ > maxFileSize_ )
           {
             checkFileSystem(); // later should take some action
             stop();
             writeToMailBox();
             fileNameCounter_++;
             // better to use temp variable as not sure if writer is still using
             // them? shouldn't be! (no time to look now)

             // also should be checking the filesystem here at path_
             std::string fileN = path_ + "/" + filen_ + "." + itoa((int)fileNameCounter_) + ".dat";
             std::string indexFileN = path_ + "/" + filen_ + "." + itoa((int)fileNameCounter_) + ".ind";

             //stream_writer_.reset( new StreamerOutputFile(fileName_+itoa(fileNameCounter_)));
             //index_writer_.reset( new StreamerOutputIndexFile(indexFileName_+itoa(fileNameCounter_)));
             stream_writer_.reset( new StreamerOutputFile(fileN));
             index_writer_.reset( new StreamerOutputIndexFile(indexFileN));

             // write out the summary line for this last file
             std::ostringstream entry;
             entry << (fileNameCounter_ - 1) << " " 
                   << fileName_
                   << " " << eventsInFile_
                   << "   " << currentFileSize_;
             files_.push_back(entry.str());
             if(fileNameCounter_!=1) closedFiles_ += ", ";
             closedFiles_ += fileName_;

             // now set the filenames
             fileName_ = path_ + "/" + filen_ + "." + itoa((int)fileNameCounter_) + ".dat";
             indexFileName_ = path_ + "/" + filen_ + "." + itoa((int)fileNameCounter_) + ".ind";

             eventsInFile_ = 0; 
             currentFileSize_ = 0;
             // write the Header for the newly openned file
             // shouldn't have to rebuild!
             InitMsgView myview(&saved_initmsg_[0]);
             uint8 psetid2[16];
             Strings hlt2;
             Strings l12;
             myview.pset(psetid2);
             myview.hltTriggerNames(hlt2);
             myview.l1TriggerNames(l12);
             uint32 desclength = myview.descLength();

             InitMsgBuilder init_message(myview.startAddress(),myview.size(),
                       myview.run(),
                       Version(myview.protocolVersion(), psetid2),
                       myview.releaseTag().c_str(),
                       hlt2,l12);

             init_message.setDescLength(desclength);
             writeHeader(init_message);
           }
                      
    // rebuild
           std::vector<bool> l1_out;
           uint8 hlt_out[10];
           eview.l1TriggerBits(l1_out);
           eview.hltTriggerBits(hlt_out);
           uint32 res = eview.reserved();
           uint32 len = eview.eventLength();

           EventMsgBuilder msg(eview.startAddress(),eview.size(),
                       eview.run(),
                       eview.event(),
                       eview.lumi(),
                       l1_out,
                       hlt_out,
                       hltsize);

           msg.setReserved(res);
           msg.setEventLength(len);

           //Write the Event Message to Streamer file
           uint64 event_offset = stream_writer_->write(msg);

           index_writer_->write(msg, event_offset);
           eventsInFile_++;
           totalEventCount_++;
           currentFileSize_ += msg.size();
  }

#include <fstream>
// to stat files and directories
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <unistd.h>
  
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
