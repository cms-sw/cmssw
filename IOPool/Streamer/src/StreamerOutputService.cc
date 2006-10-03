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
#include <stdio.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

using namespace edm;
using namespace std;

namespace edm
{

std::string itoa(int i){
        char temp[20];
        sprintf(temp,"%d",i);
        return((std::string)temp);
}

/* No one will use this CTOR anyways, we can remove it in future */
StreamerOutputService::StreamerOutputService():
 maxFileSize_(1073741824),
 maxFileEventCount_(50),
 currentFileSize_(0),
 totalEventCount_(0),
 eventsInFile_(0),
 fileNameCounter_(0),
 highWaterMark_(0.9), diskUsage_(0.0)

  {
    saved_initmsg_[0] = '\0';
    requestParamSet_ = edm::ParameterSet();
  }

 StreamerOutputService::StreamerOutputService(edm::ParameterSet const& ps):
 //maxFileSize_(ps.template getParameter<int>("maxFileSize")),
 //maxFileEventCount_(ps.template getParameter<int>("maxFileEventCount")),
 // defaulting - need numbers from Emilio
 //StreamerOutputService::StreamerOutputService():
 maxFileSize_(1073741824),
 maxFileEventCount_(50),
 currentFileSize_(0),
 totalEventCount_(0),
 eventsInFile_(0),
 fileNameCounter_(0),
 highWaterMark_(0.9), diskUsage_(0.0),
 requestParamSet_(ps)
   
  {
    saved_initmsg_[0] = '\0';
  }

void StreamerOutputService::init(std::string fileName, unsigned long maxFileSize, double highWaterMark,
                                 std::string path, std::string mpath, 
				 std::string catalog, uint32 disks,
				 InitMsgView const& view)
  { 
   maxFileSize_ = maxFileSize;
   highWaterMark_ = highWaterMark;
   path_ = path;
   mpath_ = mpath;
   filen_ = fileName;
   nLogicalDisk_   = disks;

   // create file names ( can be move to seperate method )
   std::ostringstream newFileName;
   newFileName << path_ << "/";
   catalog_        = newFileName.str() + catalog;
   lockFileName_   = newFileName.str() + "nolock";

   if (nLogicalDisk_ != 0 )
     {
       newFileName  << (fileNameCounter_ % nLogicalDisk_) << "/";
       lockFileName_   = newFileName.str() + ".lock";
       ofstream *lockFile = new ofstream(lockFileName_.c_str(), ios_base::ate | ios_base::out | ios_base::app );
       delete(lockFile);
     }

   newFileName << filen_ << "." << fileNameCounter_ ;
   fileName_      = newFileName.str() + ".dat";
   indexFileName_ = newFileName.str() + ".ind";
                                                                                                           
   statistics_  = boost::shared_ptr<edm::StreamerStatWriteService> 
     (new edm::StreamerStatWriteService(0, "-", indexFileName_, fileName_, catalog_));

   streamNindex_writer_ = boost::shared_ptr<StreamerFileWriter>(new StreamerFileWriter(fileName_, indexFileName_));

   
   //dumpInitHeader(&view);
   
   writeHeader(view);
    
   //INIT msg can be saved as INIT msg itself.

   // save the INIT message for when writing to the next file
   // that is openned
   char* pos = &saved_initmsg_[0];
   unsigned char* from = view.startAddress();
   unsigned int dsize = view.size();
   copy(from,from+dsize,pos);

   // initialize event selector
   initializeSelection(view); 

  }

void StreamerOutputService::initializeSelection(InitMsgView const& initView)
  {
  Strings triggerNameList;
  initView.hltTriggerNames(triggerNameList);

  // fake the process name (not yet available from the init message?)
  std::string processName = "HLT";

  /* ---printout the trigger names in the INIT message*/
  std::cout << ">>>>>>>>>>>Trigger names:" << std::endl;
  for(unsigned int i=0; i< triggerNameList.size(); ++i)
    std::cout<< ">>>>>>>>>>>  name = " << triggerNameList[i] << std::endl;
  /* */

  // create our event selector
  eventSelector_.reset(new EventSelector(requestParamSet_, processName,
                                         triggerNameList));
  }

StreamerOutputService::~StreamerOutputService()
  {
   // expect to do this manually at the end of a run
   // stop();   //Remove this from destructor if you want higher class to do that at its will.
              // and if stop() is ever made Public.
   writeToMailBox();

   statistics_  -> setFileSize((uint32) currentFileSize_ );
   statistics_  -> setEventCount((uint32) eventsInFile_ ); 
   statistics_  -> writeStat();

   std::ostringstream newFileName;
   newFileName << path_ << "/";
   if (nLogicalDisk_ != 0 )
     {
       newFileName  << (fileNameCounter_ % nLogicalDisk_) << "/";
// WHAT IS newFileName used for?? HEREHEREHERE
       remove( lockFileName_.c_str() );
     }
   
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

void StreamerOutputService::writeHeader(InitMsgView const& init_message)
  {
    //Write the Init Message to Streamer and Index file
    streamNindex_writer_->doOutputHeader(init_message);

    currentFileSize_ += init_message.size();

  }

void StreamerOutputService::writeEvent(EventMsgView const& eview, uint32 hltsize)
  { 

        //Check if this event meets the selection criteria, if not skip it.
        if ( ! wantsEvent(eview) ) 
            {
            //cout <<"This event is UNWANTED"<<endl; 
            return;
            }

        // check for number of events is no longer required 09/22/2006
        // if ( eventsInFile_ >= maxFileEventCount_  || currentFileSize_ >= maxFileSize_ )
	if ( currentFileSize_ >= maxFileSize_ )
           {
             checkFileSystem(); // later should take some action
             stop();
             writeToMailBox();
 
	     statistics_  -> setFileSize((uint32) currentFileSize_ );
	     statistics_  -> setEventCount((uint32) eventsInFile_ ); 
	     statistics_  -> setRunNumber((uint32) eview.run());
	     statistics_  -> writeStat();

	     fileNameCounter_++;

             string tobeclosedFile = fileName_;

             //std::cout<<" better to use temp variable as not sure if writer is still using"<<std::endl;
             // them? shouldn't be! (no time to look now)
             // writer is not using them !! - AA

             // also should be checking the filesystem here at path_
             std::ostringstream newFileName;
             newFileName << path_ << "/";
             if (nLogicalDisk_ != 0 )
               {
                 newFileName  << (fileNameCounter_ % nLogicalDisk_) << "/";
                 remove( lockFileName_.c_str() );
                 lockFileName_   = newFileName.str() + ".lock";
                 ofstream *lockFile =
                   new ofstream(lockFileName_.c_str(), ios_base::ate | ios_base::out | ios_base::app );
                 delete(lockFile);
               }
	     
             newFileName << filen_ << "." << fileNameCounter_ ;
             fileName_      = newFileName.str() + ".dat";
             indexFileName_ = newFileName.str() + ".ind";
                                               
	     statistics_  = boost::shared_ptr<edm::StreamerStatWriteService> 
	       (new edm::StreamerStatWriteService(eview.run(), "-", indexFileName_, fileName_, catalog_));
                                                            
             streamNindex_writer_.reset(new StreamerFileWriter(fileName_, indexFileName_));

             // write out the summary line for this last file
             std::ostringstream entry;
             entry << (fileNameCounter_ - 1) << " " 
                   << tobeclosedFile
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

bool StreamerOutputService::wantsEvent(EventMsgView const& eventView) 
  {
    std::vector<unsigned char> hlt_out;
    hlt_out.resize(1 + (eventView.hltCount()-1)/4);
    eventView.hltTriggerBits(&hlt_out[0]);
    /* --- print the trigger bits from the event header
    std::cout << ">>>>>>>>>>>Trigger bits:" << std::endl;
    for(int i=0; i< hlt_out.size(); ++i)
    {
      unsigned int test = static_cast<unsigned int>(hlt_out[i]);
      std::cout<< hex << ">>>>>>>>>>>  bits = " << test << " " << hlt_out[i] << std::endl;
    }
    cout << "\nhlt bits=\n(";
    for(int i=(hlt_out.size()-1); i != -1 ; --i)
       printBits(hlt_out[i]);
    cout << ")\n";
    */
    int num_paths = eventView.hltCount();
    //cout <<"num_paths: "<<num_paths<<endl;
    bool rc = (eventSelector_->wantAll() || eventSelector_->acceptEvent(&hlt_out[0], num_paths));
    //std::cout << "====================== " << std::endl;
    //std::cout << "return selector code = " << rc << std::endl;
    //std::cout << "====================== " << std::endl;
    return rc;
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
