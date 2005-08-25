
#include "IOPool/Streamer/src/TestConsumer.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <string>
#include <fstream>

using namespace edm;
using namespace std;

namespace edmtest
{

  struct Worker
  {
    Worker(const string& s);

    string filename_;
    ofstream ost_;
  };

  Worker::Worker(const string& s):
    filename_(s),
    ost_(s.c_str(),ios_base::binary | ios_base::out)
  {
    if(!ost_)
      {
	throw cms::Exception("Configuration","TestConsumer")
	  << "cannot open file " << s;
      }
  }
  
  // ----------------------------------

  TestConsumer::TestConsumer(edm::ParameterSet const& ps, 
			     edm::ProductRegistry const& reg,
			     edm::EventBuffer* buf):
    worker_(new Worker(ps.getParameter<string>("fileName"))),
    bufs_(buf)
  {
    // first write out all the product registry data into the front
    // of the output file (in text format)
  }
  
  TestConsumer::~TestConsumer()
  {
    delete worker_;
  }
  
  void TestConsumer::bufferReady()
  {
    EventBuffer::ConsumerBuffer cb(*bufs_);

    int sz = cb.size();
    worker_->ost_.write((const char*)(&sz),sizeof(int));
    worker_->ost_.write((const char*)cb.buffer(),sz);
  }

  void TestConsumer::stop()
  {
    EventBuffer::ProducerBuffer pb(*bufs_);
    pb.commit();
  }

  void TestConsumer::sendRegistry(void* buf, int len)
  {
    worker_->ost_.write((const char*)(&len),sizeof(int));
    worker_->ost_.write((const char*)buf,len);    
  }
}
