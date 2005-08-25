
#include "IOPool/Streamer/src/TestProducer.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <string>
#include <fstream>
#include <algorithm>
#include <iterator>

using namespace std;
using namespace edm;

namespace edmtestp
{
  struct Worker
  {
    Worker(const string& s);

    string filename_;
    ifstream ist_;
    TestProducer::RegBuffer regdata_;
  };

  Worker::Worker(const string& s):
    filename_(s),
    ist_(s.c_str(),ios_base::binary | ios_base::in),
    regdata_(1000*1000)
  {
    if(!ist_)
      {
	throw cms::Exception("Configuration","TestProducer")
	  << "cannot open file " << s;
      }

    int len;
    ist_.read((char*)&len,sizeof(int));
    regdata_.resize(len);
    ist_.read(&regdata_[0],len);

    if(!ist_)
      throw cms::Exception("ReadHeader","TestProducer")
	<< "Could not read the registry information from the test\n"
	<< "event stream file " << s << "\n";
  }
  
  // ----------------------------------

  TestProducer::TestProducer(edm::ParameterSet const& ps, 
			     edm::ProductRegistry const& reg,
			     edm::EventBuffer* buf):
    worker_(new Worker(ps.getParameter<string>("fileName"))),
    bufs_(buf)
  {
  }
  
  TestProducer::~TestProducer()
  {
    delete worker_;
  }

  void TestProducer::getRegistry(RegBuffer& copyhere)
  {
    RegBuffer b;
    b.reserve(worker_->regdata_.size());
    copy(worker_->regdata_.begin(),worker_->regdata_.end(),
	 back_inserter(b));
    b.swap(copyhere);
  }

  void TestProducer::stop()
  {
    // do nothing?
  }

  void TestProducer::needBuffer()
  {
    EventBuffer::ProducerBuffer pb(*bufs_);
    int len=0;
    worker_->ist_.read((char*)&len,sizeof(int));
    if(!worker_->ist_ || len==0)
      {
	pb.commit();
	return;
      }
    worker_->ist_.read((char*)pb.buffer(),len);
    pb.commit(len);
  }

}
