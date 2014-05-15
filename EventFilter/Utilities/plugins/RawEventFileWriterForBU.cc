// $Id: RawEventFileWriterForBU.cc,v 1.1.2.6 2013/03/28 14:56:53 aspataru Exp $

#include "EventFilter/Utilities/plugins/RawEventFileWriterForBU.h"
#include "EventFilter/Utilities/plugins/EvFDaqDirector.h"
#include "EventFilter/Utilities/interface/FileIO.h"
#include "EventFilter/Utilities/interface/JSONSerializer.h"

#include "FWCore/Utilities/interface/Adler32Calculator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <signal.h>
#include <boost/filesystem/fstream.hpp>

//TODO:get run directory information from DaqDirector

RawEventFileWriterForBU* RawEventFileWriterForBU::instance = 0;

RawEventFileWriterForBU::RawEventFileWriterForBU(edm::ParameterSet const& ps):
      // default to .5ms sleep per event
       microSleep_(ps.getUntrackedParameter<int>("microSleep", 0))
       //debug_(ps.getUntrackedParameter<bool>("debug", False))
{

  //per-file JSD and FastMonitor
  rawJsonDef_.setDefaultGroup("legend");
  rawJsonDef_.addLegendItem("NEvents","integer",DataPointDefinition::SUM);

  perFileEventCount_.setName("NEvents");

  fileMon_ = new FastMonitor(&rawJsonDef_,false);
  fileMon_->registerGlobalMonitorable(&perFileEventCount_,false,nullptr);
  fileMon_->commit(nullptr);

  //per-lumi JSD and FastMonitor
  eolJsonDef_.setDefaultGroup("legend");
  eolJsonDef_.addLegendItem("NEvents","integer",DataPointDefinition::SUM);
  eolJsonDef_.addLegendItem("NFiles","integer",DataPointDefinition::SUM);
  eolJsonDef_.addLegendItem("TotalEvents","integer",DataPointDefinition::SUM);

  perLumiEventCount_.setName("NEvents");
  perLumiFileCount_.setName("NFiles");
  perLumiTotalEventCount_.setName("TotalEvents");

  lumiMon_ = new FastMonitor(&eolJsonDef_,false);
  lumiMon_->registerGlobalMonitorable(&perLumiEventCount_,false,nullptr);
  lumiMon_->registerGlobalMonitorable(&perLumiFileCount_,false,nullptr);
  lumiMon_->registerGlobalMonitorable(&perLumiTotalEventCount_,false,nullptr);
  lumiMon_->commit(nullptr);


  //per-run JSD and FastMonitor
  eorJsonDef_.setDefaultGroup("legend");
  eorJsonDef_.addLegendItem("NEvents","integer",DataPointDefinition::SUM);
  eorJsonDef_.addLegendItem("NFiles","integer",DataPointDefinition::SUM);
  eorJsonDef_.addLegendItem("NLumis","integer",DataPointDefinition::SUM);

  perRunEventCount_.setName("NEvents");
  perRunFileCount_.setName("NFiles");
  perRunLumiCount_.setName("NLumis");
 
  runMon_ = new FastMonitor(&eorJsonDef_,false);
  runMon_->registerGlobalMonitorable(&perRunEventCount_,false,nullptr);
  runMon_->registerGlobalMonitorable(&perRunFileCount_,false,nullptr);
  runMon_->registerGlobalMonitorable(&perRunLumiCount_,false,nullptr);
  runMon_->commit(nullptr);

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

}

RawEventFileWriterForBU::~RawEventFileWriterForBU()
{
  delete fileMon_;
  delete lumiMon_;
  delete runMon_;
}

void RawEventFileWriterForBU::doOutputEvent(FRDEventMsgView const& msg)
{
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
  perLumiTotalEventCount_.value()++;
  perFileEventCount_.value()++;

  //  cms::Adler32((const char*) msg.startAddress(), msg.size(), adlera_, adlerb_);
}

void RawEventFileWriterForBU::doOutputEventFragment(unsigned char* dataPtr, unsigned long dataSize)
{

  throw cms::Exception("RawEventFileWriterForBU", "doOutputEventFragment")
    << "Unsupported output mode ";

  //cms::Adler32((const char*) dataPtr, dataSize, adlera_, adlerb_);
}

void RawEventFileWriterForBU::initialize(std::string const& destinationDir, std::string const& name, int ls)
{
  std::string oldFileName = fileName_;
  fileName_ = name;
  destinationDir_ = destinationDir;

  if (!writtenJSDs_) {

    std::stringstream ss;
    ss << destinationDir_ << "/jsd";
    mkdir(ss.str().c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);

    std::string rawJSDName = ss.str()+"/rawData.jsd";
    std::string eolJSDName = ss.str()+"/EoLS.jsd";
    std::string eorJSDName = ss.str()+"/EoR.jsd";

    fileMon_->setDefPath(rawJSDName);
    lumiMon_->setDefPath(eolJSDName);
    runMon_->setDefPath(eorJSDName);

    struct stat   fstat;
    if (stat (rawJSDName.c_str(), &fstat) != 0) {
      std::string content;
      JSONSerializer::serialize(&rawJsonDef_,content);
      FileIO::writeStringToFile(rawJSDName, content);
    }

    if (stat (eolJSDName.c_str(), &fstat) != 0) {
      std::string content;
      JSONSerializer::serialize(&eolJsonDef_,content);
      FileIO::writeStringToFile(eolJSDName, content);
    }

    if (stat (eorJSDName.c_str(), &fstat) != 0) {
      std::string content;
      JSONSerializer::serialize(&eorJsonDef_,content);
      FileIO::writeStringToFile(eorJSDName, content);
    }

    writtenJSDs_=true;

  }
  closefd();
  outfd_ = open(fileName_.c_str(), O_WRONLY | O_CREAT,  S_IRWXU | S_IRWXG | S_IRWXO);
   edm::LogInfo("RawEventFileWriterForBU") << " opened " << fileName_;
  if(outfd_ < 0) { //attention here... it may happen that outfd_ is *not* set (e.g. missing initialize call...)
    throw cms::Exception("RawEventFileWriterForBU","initialize")
      << "Error opening FED Raw Data event output file: " << name
      << ": " << strerror(errno) << "\n";
  }

  //move old file to done directory
  if (!oldFileName.empty()) {


    //move raw file from open to run directory
    rename(oldFileName.c_str(),(destinationDir_+oldFileName.substr(oldFileName.rfind("/"))).c_str());

    //create equivalent JSON file

    std::stringstream ss;
    //TODO:fix this to use DaqDirector convention and better extension replace
    boost::filesystem::path source(oldFileName);
    std::string path = source.replace_extension(".jsn").string();

    fileMon_->snap(ls);
    fileMon_->outputFullJSON(path, ls, false);
    fileMon_->discardCollected(ls);

    //move the json file from open
    rename(path.c_str(),(destinationDir_+path.substr(path.rfind("/"))).c_str());

    edm::LogInfo("RawEventFileWriterForBU") << "Wrote JSON input file: " << path 
					    << " with perFileEventCount = " << perFileEventCount_.value();

  }

  perFileEventCount_.value() = 0;
  perLumiFileCount_.value()++;


  adlera_ = 1;
  adlerb_ = 0;
}

void RawEventFileWriterForBU::endOfLS(int ls)
{
  closefd();
  lumiMon_->snap(ls);

  std::ostringstream ostr;

  if (run_==-1) makeRunPrefix(destinationDir_);

  ostr << destinationDir_ << "/"<< runPrefix_ << "_ls" << std::setfill('0') << std::setw(4) << ls << "_EoLS" << ".jsn";
  outfd_ = open(ostr.str().c_str(), O_WRONLY | O_CREAT,  S_IWUSR | S_IRUSR | S_IWGRP | S_IRGRP | S_IWOTH | S_IROTH);
  closefd();

  std::string path = ostr.str();
  lumiMon_->outputFullJSON(path, ls);
  lumiMon_->discardCollected(ls);

  perRunEventCount_.value() += perLumiEventCount_.value();
  perRunFileCount_.value() += perLumiFileCount_.value();
  perRunLumiCount_.value() += 1;

  perLumiEventCount_ = 0;
  perLumiFileCount_ = 0;
  perLumiTotalEventCount_ = 0;
}

//runs on SIGINT and terminates the process
void RawEventFileWriterForBU::handler(int s){
  printf("Caught signal %d. Writing EOR file!\n",s);
  if (destinationDir_.size() > 0)
    {
      // CREATE EOR file
      if (run_==-1) makeRunPrefix(destinationDir_);
      std::string path = destinationDir_ + "/" + runPrefix_ + "_ls0000_EoR.jsn";
      runMon_->snap(0);
      runMon_->outputFullJSON(path, 0);
    }
  _exit(0);
}

//TODO:get from DaqDirector !
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
