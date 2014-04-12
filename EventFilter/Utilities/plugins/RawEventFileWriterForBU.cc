// $Id: RawEventFileWriterForBU.cc,v 1.1.2.6 2013/03/28 14:56:53 aspataru Exp $

#include "RawEventFileWriterForBU.h"
#include "FWCore/Utilities/interface/Adler32Calculator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "EventFilter/Utilities/interface/FileIO.h"
#include "EventFilter/Utilities/plugins/EvFDaqDirector.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <signal.h>


RawEventFileWriterForBU* RawEventFileWriterForBU::instance = 0;

void RawEventFileWriterForBU::handler(int s){
  printf("Caught signal %d. Writing EOR file!\n",s);
  if (destinationDir_.size() > 0)
    {
      // CREATE EOR file
      
      if (run_==-1) makeRunPrefix(destinationDir_);

      std::string path = destinationDir_ + "/" + runPrefix_ + "_ls0000_EoR.jsn";
      std::string output = "EOR";
      FileIO::writeStringToFile(path, output);
    }
  _exit(0);
}

RawEventFileWriterForBU::RawEventFileWriterForBU(edm::ParameterSet const& ps): lumiMon_(0), outfd_(0),
       jsonDefLocation_(ps.getUntrackedParameter<std::string>("jsonDefLocation","")),
      // default to .5ms sleep per event
       microSleep_(ps.getUntrackedParameter<int>("microSleep", 0))
{
  //  initialize(ps.getUntrackedParameter<std::string>("fileName", "testFRDfile.dat"));
  perLumiEventCount_ = 0;
  // set names of the variables to be matched with JSON Definition
  perLumiEventCount_.setName("NEvents");

  // create a FastMonitor using monitorable parameters and a path to a JSON Definition file
  std::string defGroup = "data";
  lumiMon_ = new FastMonitor(jsonDefLocation_,defGroup,false);
  lumiMon_->registerGlobalMonitorable(&perLumiEventCount_,false,nullptr);
  lumiMon_->commit(nullptr);


  perFileEventCount_.value() = 0;
  perFileEventCount_.setName("NEvents");
  // create a FastMonitor using monitorable parameters and a path to a JSON Definition file
  perFileMon_ = new FastMonitor(jsonDefLocation_,defGroup,false);
  perFileMon_->registerGlobalMonitorable(&perFileEventCount_,false,nullptr);
  perFileMon_->commit(nullptr);
  instance = this;

  // SIGINT Handler
  struct sigaction sigIntHandler;
  sigIntHandler.sa_handler = RawEventFileWriterForBU::staticHandler;
  sigemptyset(&sigIntHandler.sa_mask);
  sigIntHandler.sa_flags = 0;
  sigaction(SIGINT, &sigIntHandler, NULL);

}

RawEventFileWriterForBU::RawEventFileWriterForBU(std::string const& fileName)
{
  //  initialize(fileName);

}

RawEventFileWriterForBU::~RawEventFileWriterForBU()
{
  //  ost_->close();
  if (lumiMon_ != 0)
    delete lumiMon_;
  if (perFileMon_ != 0)
    delete perFileMon_;
}

void RawEventFileWriterForBU::doOutputEvent(FRDEventMsgView const& msg)
{
  //  ost_->write((const char*) msg.startAddress(), msg.size());
  //  if (ost_->fail()) {
  ssize_t retval =  write(outfd_,(void*)msg.startAddress(), msg.size());
  if((unsigned)retval!= msg.size()){
    throw cms::Exception("RawEventFileWriterForBU", "doOutputEvent")
      << "Error writing FED Raw Data event data to "
      << fileName_ << ".  Possibly the output disk "
      << "is full?" << std::endl;
  }

  // throttle event output
  usleep(microSleep_);

  perLumiEventCount_.value()++;
  perFileEventCount_.value()++;

  //   ost_->flush();
  //   if (ost_->fail()) {
  //     throw cms::Exception("RawEventFileWriterForBU", "doOutputEvent")
  //       << "Error writing FED Raw Data event data to "
  //       << fileName_ << ".  Possibly the output disk "
  //       << "is full?" << std::endl;
  //   }

  //  cms::Adler32((const char*) msg.startAddress(), msg.size(), adlera_, adlerb_);
}

void RawEventFileWriterForBU::doFlushFile()
{
  ost_->flush();
  if (ost_->fail()) {
    throw cms::Exception("RawEventFileWriterForBU", "doOutputEvent")
      << "Error writing FED Raw Data event data to "
      << fileName_ << ".  Possibly the output disk "
      << "is full?" << std::endl;
  }
}

void RawEventFileWriterForBU::doOutputEventFragment(unsigned char* dataPtr,
						    unsigned long dataSize)
{
  ost_->write((const char*) dataPtr, dataSize);
  if (ost_->fail()) {
    throw cms::Exception("RawEventFileWriterForBU", "doOutputEventFragment")
      << "Error writing FED Raw Data event data to "
      << fileName_ << ".  Possibly the output disk "
      << "is full?" << std::endl;
  }

  ost_->flush();
  if (ost_->fail()) {
    throw cms::Exception("RawEventFileWriterForBU", "doOutputEventFragment")
      << "Error writing FED Raw Data event data to "
      << fileName_ << ".  Possibly the output disk "
      << "is full?" << std::endl;
  }

  cms::Adler32((const char*) dataPtr, dataSize, adlera_, adlerb_);
}

void RawEventFileWriterForBU::initialize(std::string const& destinationDir, std::string const& name, int ls)
{
  std::string oldFileName = fileName_;
  fileName_ = name;
  destinationDir_ = destinationDir;
  if(outfd_!=0){
    close(outfd_);
    outfd_=0;
  }
  outfd_ = open(fileName_.c_str(), O_WRONLY | O_CREAT,  S_IRWXU);
  if(outfd_ <= 0) { //attention here... it may happen that outfd_ is *not* set (e.g. missing initialize call...)
    throw cms::Exception("RawEventFileWriterForBU","initialize")
      << "Error opening FED Raw Data event output file: " << name
      << ": " << strerror(errno) << "\n";
  }
  ost_.reset(new std::ofstream(name.c_str(), std::ios_base::binary | std::ios_base::out));

  //move old file to done directory
  if (!oldFileName.empty()) {
    //rename(oldFileName.c_str(),destinationDir_.c_str());

    perFileMon_->snap(false, "",ls);

    std::stringstream ss;
    ss << destinationDir_ << "/" << oldFileName.substr(oldFileName.rfind("/") + 1, oldFileName.size() - oldFileName.rfind("/") - 5) << ".jsn";
    std::string path = ss.str();

    perFileMon_->outputFullJSON(path, ls);
    perFileMon_->discardCollected(ls);
    //now that the json file is there, move the raw file
    int fretval = rename(oldFileName.c_str(),(destinationDir_+oldFileName.substr(oldFileName.rfind("/"))).c_str());
    // if (debug_)
    edm::LogInfo("RawEventFileWriterForBU") << " tried move " << oldFileName << " to " << destinationDir_
					    << " status "  << fretval << " errno " << strerror(errno);
    
    edm::LogInfo("RawEventFileWriterForBU") << "Wrote JSON input file: " << path 
					    << " with perFileEventCount = " << perFileEventCount_.value();

    perFileEventCount_.value() = 0;

  }


  if (!ost_->is_open()) {
    throw cms::Exception("RawEventFileWriterForBU","initialize")
      << "Error opening FED Raw Data event output file: " << name << "\n";
  }

  adlera_ = 1;
  adlerb_ = 0;
}

void RawEventFileWriterForBU::endOfLS(int ls)
{
  //writing empty EoLS file (will be filled with information)
  //take snapshot of the monitored data into it

  lumiMon_->snap(false,"",ls);

  std::ostringstream ostr;

  if (run_==-1) makeRunPrefix(destinationDir_);

  ostr << destinationDir_ << "/"<< runPrefix_ << "_ls" << std::setfill('0') << std::setw(4) << ls << "_EoLS" << ".jsn";
  int outfd_ = open(ostr.str().c_str(), O_WRONLY | O_CREAT,  S_IRWXU | S_IRWXG | S_IRWXO);
  if(outfd_!=0){close(outfd_); outfd_=0;}

  std::string path = ostr.str();
  // serialize the DataPoint and output it
  lumiMon_->outputFullJSON(path, ls);

  perLumiEventCount_ = 0;
}

void RawEventFileWriterForBU::makeRunPrefix(std::string const& destinationDir)
{
  //dirty hack: extract run number from destination directory
  std::string::size_type pos = destinationDir.find("run");
  std::string run = destinationDir.substr(pos+3);
  run_=atoi(run.c_str());
  std::stringstream ss;
  ss << "run" << std::setfill('0') << std::setw(6) << run_;
  runPrefix_ = ss.str();
}
