// $Id: MTRawEventFileWriterForBU.cc,v 1.1.2.7 2013/01/16 17:47:37 aspataru Exp $

#include "EventFilter/Utilities/plugins/MTRawEventFileWriterForBU.h"
#include "FWCore/Utilities/interface/Adler32Calculator.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <assert.h>
#include <boost/tokenizer.hpp>


namespace fwriter {
  class EventContainer {
    public:
      EventContainer(unsigned int evtBufSize)
      {
	writtenSize_=0;
	evtBufSize_=evtBufSize;
	data_.reset(new unsigned char [evtBufSize]);
	shared_mode_=false;
      }
      EventContainer()
      {
	writtenSize_=0;
	evtBufSize_=0;
	shared_data_.reset();
	shared_mode_=true;
      }
      ~EventContainer() {}

      bool sharedMode() {return shared_mode_;}
      unsigned int getSize() const {return writtenSize_;}
      unsigned int getBufSize() const {return evtBufSize_;}
      unsigned char* getBuffer() const {return data_.get();}
      boost::shared_array<unsigned char> * getSharedBuffer() {return & shared_data_;}

      void putNewEvent(unsigned char* addr, unsigned int size) {
	if (size>evtBufSize_) {
	  data_.reset(new unsigned char[size]);
	  evtBufSize_=size;
	}
	memcpy(data_.get(),addr,size);
	writtenSize_=size;
      }

      void putNewEvent(boost::shared_array<unsigned char> & msg) {
	shared_data_ = msg;
      }

    private:
      unsigned int writtenSize_;
      unsigned int evtBufSize_;
      std::auto_ptr<unsigned char> data_;
      boost::shared_array<unsigned char> shared_data_;
      bool shared_mode_;
  };
}

MTRawEventFileWriterForBU::MTRawEventFileWriterForBU(edm::ParameterSet const& ps):
	numWriters_(ps.getUntrackedParameter<unsigned int>("numWriters",1))
	,eventBufferSize_(ps.getUntrackedParameter<unsigned int>("eventBufferSize",30))
	,sharedMode_(ps.getUntrackedParameter<bool>("sharedMode",true))
	,lumiSubdirectoriesMode_(ps.getUntrackedParameter<bool>("lumiSubdirectoriesMode",true))
	,debug_(ps.getUntrackedParameter<bool>("debug",false))
	//,finishAfterLS_(ps.getUntrackedParameter<int>,-1)
{
	for (unsigned int i=0;i<eventBufferSize_;i++) {
		if (!sharedMode_)
			EventPool.push_back(new fwriter::EventContainer(1048576));
		else
			EventPool.push_back(new fwriter::EventContainer());
		freeIds.push_back(i);
	}
	fileHeader_= new unsigned char[1024*1024];

	perLumiEventCount_ = 0;
	// set names of the variables to be matched with JSON Definition
	perLumiEventCount_.setName("NEvents");
        std::string defGroup = "data";
      	lumiMon_.reset(new FastMonitor("$CMSSW_BASE/src/EventFilter/Utilities/plugins/budef.jsd",defGroup,false));
        lumiMon_->registerGlobalMonitorable(&perLumiEventCount_,false,nullptr);
        lumiMon_->commit(nullptr);


}


MTRawEventFileWriterForBU::~MTRawEventFileWriterForBU()
{
  finishThreads();
  while (!perFileCounters_.empty()) {
  	  delete perFileCounters_.back();
  	perFileCounters_.pop_back();
    }
}

void MTRawEventFileWriterForBU::doOutputEvent(FRDEventMsgView const& msg)
{
  if (sharedMode_) return;
  queueEvent((const char*)msg.startAddress(), msg.size());
}


void MTRawEventFileWriterForBU::doOutputEvent(boost::shared_array<unsigned char> & msg)
{
  if (!sharedMode_) return;
  queueEvent(msg);
}

void MTRawEventFileWriterForBU::doFlushFile()
{
  //not implemented
}

void MTRawEventFileWriterForBU::doOutputEventFragment(unsigned char* dataPtr,
                                               unsigned long dataSize)
{
  queueEvent((const char*) dataPtr, dataSize);

  /*
  ost_->flush();
  if (ost_->fail()) {
    throw cms::Exception("RawEventFileWriterForBU", "doOutputEventFragment")
      << "Error writing FED Raw Data event data to "
      << fileName_ << ".  Possibly the output disk "
      << "is full?" << std::endl;
  }
  */

  //cms::Adler32((const char*) dataPtr, dataSize, adlera_, adlerb_);
}

void MTRawEventFileWriterForBU::initialize(std::string const& destinationDir, std::string const& name, int ls)
{

  destinationDir_ = destinationDir+"/";
  if (lumiSubdirectoriesMode_) {
    std::ostringstream lsdir;
    lsdir << "ls"  << std::setfill('0') << std::setw(6) << ls << "/";
    lumiSectionSubDir_ = lsdir.str();
    mkdir((destinationDir_+lumiSectionSubDir_).c_str(),0755);
  }
  else lumiSectionSubDir_="";

  std::string fileBase=name;
  std::string fileSuffix;

  boost::char_separator<char> sep(".");
  boost::tokenizer<boost::char_separator<char>> tokens(name, sep);

  fileBase=*tokens.begin();
  for (auto tok_iter = tokens.begin(); tok_iter != tokens.end(); ++tok_iter) {
    fileSuffix=*tok_iter;
  }
  finishThreads();
  dispatchThreads(fileBase,numWriters_,fileSuffix,ls);
}

void MTRawEventFileWriterForBU::endOfLS(int ls)
{
	finishThreads();
	//writing empty EoLS file (will be filled with information)
	// MARK! BU EOL json OLD!!!

	// create a DataPoint object and take a snapshot of the monitored data into it

	lumiMon_->snap(false,"",ls);

	std::ostringstream ostr;
	ostr << destinationDir_ << "/EoLS_" << std::setfill('0') << std::setw(6) << ls << ".json";
	int outfd_ = open(ostr.str().c_str(), O_WRONLY | O_CREAT,  S_IRWXU);
	if(outfd_!=0){close(outfd_); outfd_=0;}

	std::string path = ostr.str();
	// serialize the DataPoint and output it
	lumiMon_->outputFullJSON(path, ls);

	perLumiEventCount_ = 0;


}

void MTRawEventFileWriterForBU::queueEvent(const char* buffer,unsigned long size)
{

#ifdef linux
  bool queuing = false;
  unsigned int freeId = 0xffff;
  while (!queuing) {
    queue_lock.lock();
    if (freeIds.size()) {
      freeId = freeIds.front();
      freeIds.pop_front();
      queuing = true;
    }
    queue_lock.unlock();
    if (!queuing) usleep(100000);
  }
  assert(freeId!=0xff);
  EventPool[freeId]->putNewEvent((unsigned char*)buffer,size);

  queue_lock.lock();
  queuedIds.push_back(freeId);
  perLumiEventCount_.value()++;
  queue_lock.unlock();
#endif
}


void MTRawEventFileWriterForBU::queueEvent(boost::shared_array<unsigned char>& msg)
{

#ifdef linux
  bool queuing = false;
  unsigned int freeId = 0xffff;
  while (!queuing) {
    queue_lock.lock();
    if (freeIds.size()) {
      freeId = freeIds.front();
      freeIds.pop_front();
      queuing = true;
    }
    queue_lock.unlock();
    if (!queuing) usleep(100000);
  }
  assert(freeId!=0xff);
  EventPool[freeId]->putNewEvent(msg);

  queue_lock.lock();
  queuedIds.push_back(freeId);
  perLumiEventCount_.value()++;
  queue_lock.unlock();
#endif
}


void MTRawEventFileWriterForBU::dispatchThreads(std::string fileBase, unsigned int instances, std::string suffix, unsigned int ls)
{
#ifdef linux
  close_flag_=false;
  //v_adlera_(numWriters_,1);
  //v_adlerb_(numWriters_,0);
  for (unsigned int i=0;i<instances;i++)
  {
    std::ostringstream instanceName;
    instanceName << fileBase << "_" << i;
    if (suffix.size())
      instanceName << "." << suffix;

    // populate perFileCounters
    IntJ* currentVal = new IntJ();
    *currentVal = 0;
    currentVal->setName("NEvents");
    perFileCounters_.push_back(currentVal);

    // create per-file FastMonitor instances using a path to a JSON Definition file
    std::string defGroup = "data";
    std::unique_ptr<FastMonitor> thisMon;
    thisMon.reset(new FastMonitor("$CMSSW_BASE/src/EventFilter/Utilities/plugins/budef.jsd",defGroup,false);
    thisMon->registerGlobalMonitorable(&perFileCounters_[i],false,nullptr);
    thisMon->commit(nullptr);
    perFileMonitors_.push_back(std::move(thisMon));

    writers.push_back(std::auto_ptr<std::thread>(new std::thread(&MTRawEventFileWriterForBU::threadRunner,this,instanceName.str(),i,ls)));
  }
#endif
}


void MTRawEventFileWriterForBU::threadRunner(std::string fileName,unsigned int instance,unsigned int ls)
{
#ifdef linux
  //new file..
  if (debug_)
    std::cout << "opening file for writing " << fileName.c_str() << std::endl;
  int outfd_ = open(fileName.c_str(), O_WRONLY | O_CREAT,  S_IRWXU);
  if(outfd_ == -1) {
    throw cms::Exception("RawEventFileWriterForBU","initialize")
      << "Error opening FED Raw Data event output file: " << fileName 
      << ": " << strerror(errno) << "\n";
  }
  std::auto_ptr<std::ofstream> ost_;
  ost_.reset(new std::ofstream(fileName.c_str(), std::ios_base::binary | std::ios_base::out));

  if (!ost_->is_open()) {
    throw cms::Exception("RawEventFileWriterForBU","initialize")
      << "Error opening FED Raw Data event output file: " << fileName << "\n";
  }
  //prepare header
  /*
  memset ((void*)fileHeader_,0,1024*1024);
  fileHeader_[0]=3;//version
  ost_->write((const char*)fileHeader_,1024*1024);
  */
  //event writing loop
  while (1) {
    queue_lock.lock();

    if (!queuedIds.size()) {
      if (!close_flag_) {
	queue_lock.unlock();
	usleep(100000);//todo:use timed cond wait
	continue;
      }
    }
    if (close_flag_) {
      queue_lock.unlock();
      break;
    }
    //take next event
    unsigned int qid = queuedIds.back();
    queuedIds.pop_back();
    queue_lock.unlock();
    if (!EventPool[qid]->sharedMode()) {
      ost_->write((const char*) EventPool[qid]->getBuffer(), EventPool[qid]->getSize());
    }
    else {
      boost::shared_array<unsigned char> * sharedBuf = EventPool[qid]->getSharedBuffer();
      FRDEventMsgView frd((*sharedBuf).get());
      ost_->write((const char*) frd.startAddress(),frd.size());
      sharedBuf->reset();//release reference
    }
    if (ost_->fail()) {
      //todo:signal to main thread
      throw cms::Exception("RawEventFileWriterForBU", "doOutputEventFragment")
	<< "Error writing FED Raw Data event data to "
	<< fileName_ << ".  Possibly the output disk "
        << "is full?" << std::endl;
    }

    queue_lock.lock();
    freeIds.push_back(qid);
    perFileCounters_[instance]->value()++;
    queue_lock.unlock();
  }
  //flush and close file
  ost_->flush();
  if (ost_->fail()) {
    throw cms::Exception("RawEventFileWriterForBU", "doOutputEventFragment")
      << "Error writing FED Raw Data event data to "
      << fileName_ << ".  Possibly the output disk "
      << "is full?" << std::endl;
  }
  ost_.reset();
  if(outfd_!=0){ close(outfd_); outfd_=0;}
  
  //move file to destination dir
  int fretval = rename(fileName.c_str(),(destinationDir_+lumiSectionSubDir_+fileName.substr(fileName.rfind("/"))).c_str());
  if (debug_)
    std::cout << " tried move " << fileName << " to " << destinationDir_+lumiSectionSubDir_
              << " status "  << fretval << " errno " << strerror(errno) << std::endl;

  // MARK! BU per-file json OLD!!!
  perFileMonitors_[instance]->snap(false,"",ls);

  std::stringstream ss;
  ss << destinationDir_ << lumiSectionSubDir_ << fileName.substr(fileName.rfind("/") + 1, fileName.size() - fileName.rfind("/") - 5) << ".jsn";
  string path = ss.str();

  perFileMonitors_[instance]->outputFullHistoDataPoint(path, ls);
  if (debug_)
	std::cout << "Wrote JSON input file: " << path << std::endl;

  perFileCounters_[instance]->value() = 0;

#endif
}
