
#include "IOPool/Streamer/src/TestConsumer.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "boost/shared_ptr.hpp"

#include <string>
#include <cstring>
#include <fstream>
#include <sstream>

using namespace edm;

namespace edmtest
{
  typedef boost::shared_ptr<std::ofstream> OutPtr;
  typedef std::vector<char> SaveArea;

  std::string makeFileName(const std::string& base, int num)
  {
    std::ostringstream ost;
    ost << base << num << ".dat";
    return ost.str();
  }

  OutPtr makeFile(const std::string name,int num)
  {
    OutPtr p(new std::ofstream(makeFileName(name,num).c_str(),
			  std::ios_base::binary | std::ios_base::out));

    if(!(*p))
      {
	throw edm::Exception(errors::Configuration,"TestConsumer")
	  << "cannot open file " << name;
      }

    return p;
  }

  struct Worker
  {
    Worker(const std::string& s, int m);

    std::string filename_;
    int file_num_;
    int cnt_;
    int max_;
    OutPtr ost_; 
    SaveArea reg_;

    void checkCount();
    void saveReg(void* buf, int len);
    void writeReg();
  };

  Worker::Worker(const std::string& s,int m):
    filename_(s),
    file_num_(),
    cnt_(0),
    max_(m),
    ost_(makeFile(filename_,file_num_))
  {
  }
  
  void Worker::checkCount()
  {
    if(cnt_!=0 && (cnt_%max_) == 0)
      {
	++file_num_;
	ost_ = makeFile(filename_,file_num_);
	writeReg();
      }
    ++cnt_;

  }

  void Worker::writeReg()
  {
    if(!reg_.empty())
      {
	int len = reg_.size();
	ost_->write((const char*)(&len),sizeof(int));
	ost_->write((const char*)&reg_[0],len);
      }
  }

  void Worker::saveReg(void* buf, int len)
  {
    reg_.resize(len);
    memcpy(&reg_[0],buf,len);
  }


  // ----------------------------------

  TestConsumer::TestConsumer(edm::ParameterSet const& ps, 
			     edm::EventBuffer* buf):
    worker_(new Worker(ps.getParameter<std::string>("fileName"),
		       ps.getUntrackedParameter<int>("numPerFile",1<<31))),
    bufs_(buf)
  {
    // first write out all the product registry data into the front
    // of the output file (in text format)
  }
  
  TestConsumer::~TestConsumer()
  {
  }
  
  void TestConsumer::bufferReady()
  {
    worker_->checkCount();

    EventBuffer::ConsumerBuffer cb(*bufs_);

    int sz = cb.size();
    worker_->ost_->write((const char*)(&sz),sizeof(int));
    worker_->ost_->write((const char*)cb.buffer(),sz);

  }

  void TestConsumer::stop()
  {
    EventBuffer::ProducerBuffer pb(*bufs_);
    pb.commit();
  }

  void TestConsumer::sendRegistry(void* buf, int len)
  {
    worker_->saveReg(buf,len);
    worker_->writeReg();
  }
}
