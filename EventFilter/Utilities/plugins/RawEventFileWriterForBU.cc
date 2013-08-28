// $Id: RawEventFileWriterForBU.cc,v 1.1.2.6 2013/03/28 14:56:53 aspataru Exp $

#include "RawEventFileWriterForBU.h"
#include "FWCore/Utilities/interface/Adler32Calculator.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "EventFilter/Utilities/interface/FileIO.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <signal.h>


RawEventFileWriterForBU* RawEventFileWriterForBU::instance = 0;

void RawEventFileWriterForBU::handler(int s){
	printf("Caught signal %d. Writing EOR file!\n",s);
	if (destinationDir_.size() > 0)
	{
		// DELETE RUN DIR ON BU
		/*
		std::string rmCommand = "rm -rf ";
		std::string runDir = destinationDir_.substr(0, destinationDir_.size() - (destinationDir_.size() - destinationDir_.rfind("/")));
		rmCommand += runDir;
		int rc = system(rmCommand.c_str());
		printf("Executed command %s. Return code is %d\n", rmCommand.c_str(), rc);
		*/
		// CREATE EOR file
		string path = destinationDir_ + "/" + "EoR.jsd";
		string output = "EOR";
		FileIO::writeStringToFile(path, output);
	}
	exit(0);
}

RawEventFileWriterForBU::RawEventFileWriterForBU(edm::ParameterSet const& ps): lumiMon_(0), outfd_(0),
	jsonDefLocation_(ps.getUntrackedParameter<string>("jsonDefLocation","")),
	// default to .5ms sleep per event
	microSleep_(ps.getUntrackedParameter<int>("microSleep", 0))
{
  //  initialize(ps.getUntrackedParameter<std::string>("fileName", "testFRDfile.dat"));
	perLumiEventCount_ = 0;
	// set names of the variables to be matched with JSON Definition
	perLumiEventCount_.setName("NEvents");

	// create a vector of all monitorable parameters to be passed to the monitor
	vector<JsonMonitorable*> lumiMonParams;
	lumiMonParams.push_back(&perLumiEventCount_);

	// create a DataPointMonitor using vector of monitorable parameters and a path to a JSON Definition file
	lumiMon_ = new DataPointMonitor(lumiMonParams, jsonDefLocation_);


	perFileEventCount_.value() = 0;
	perFileEventCount_.setName("NEvents");

    // create a vector of all monitorable parameters to be passed to the monitor
    vector<JsonMonitorable*> fileMonParams;
    fileMonParams.push_back(&perFileEventCount_);

    perFileMon_ = new DataPointMonitor(fileMonParams, jsonDefLocation_);
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
  if(retval!= msg.size()){
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
  if(outfd_ == -1) {
    throw cms::Exception("RawEventFileWriterForBU","initialize")
      << "Error opening FED Raw Data event output file: " << name
      << ": " << strerror(errno) << "\n";
  }
  ost_.reset(new std::ofstream(name.c_str(), std::ios_base::binary | std::ios_base::out));

  //move old file to done directory
  if (!oldFileName.empty()) {
	  //rename(oldFileName.c_str(),destinationDir_.c_str());
	  int fretval = rename(oldFileName.c_str(),(destinationDir_+oldFileName.substr(oldFileName.rfind("/"))).c_str());
	  // if (debug_)
	  std::cout << " tried move " << oldFileName << " to " << destinationDir_
			  << " status "  << fretval << " errno " << strerror(errno) << std::endl;

	  DataPoint dp;
	  perFileMon_->snap(dp);
	  string output;
	  JSONSerializer::serialize(&dp, output);
	  std::stringstream ss;
	  ss << destinationDir_ << "/" << oldFileName.substr(oldFileName.rfind("/") + 1, oldFileName.size() - oldFileName.rfind("/") - 5) << ".jsn";
	  string path = ss.str();
	  FileIO::writeStringToFile(path, output);

	  std::cout << "Wrote JSON input file: " << path << " with perFileEventCount = " << perFileEventCount_.value() << std::endl;

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
	// create a DataPoint object and take a snapshot of the monitored data into it
	DataPoint dp;
	lumiMon_->snap(dp);

	std::ostringstream ostr;
	ostr << destinationDir_ << "/EoLS_" << std::setfill('0') << std::setw(4) << ls << ".jsn";
	int outfd_ = open(ostr.str().c_str(), O_WRONLY | O_CREAT,  S_IRWXU);
	if(outfd_!=0){close(outfd_); outfd_=0;}

	// serialize the DataPoint and output it
	string output;
	JSONSerializer::serialize(&dp, output);

	string path = ostr.str();
	FileIO::writeStringToFile(path, output);

	perLumiEventCount_ = 0;
}
