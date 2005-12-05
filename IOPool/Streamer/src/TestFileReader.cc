
#include "IOPool/Streamer/interface/TestFileReader.h"
#include "IOPool/Streamer/interface/BufferArea.h"
#include "IOPool/Streamer/interface/HLTInfo.h"
#include "IOPool/Streamer/interface/ClassFiller.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "boost/bind.hpp"
#include "boost/scoped_array.hpp"

#include <algorithm>
#include <iterator>

using namespace std;
using namespace edm;

namespace edmtestp
{  

  namespace 
  {
    struct BufHelper
    {
      explicit BufHelper(int len): buf_(new char(len)) { }
      ~BufHelper() { delete [] buf_; }
      char* release() { char* rc = buf_; buf_=0; return rc; }
      char* get() const { return buf_; }

    private:
      BufHelper(const BufHelper& ) { }
      BufHelper& operator=(const BufHelper&) { return *this; }

      char* buf_;
    };
  }

  // ----------------------------------

  TestFileReader::TestFileReader(const std::string& filename,
				 edm::EventBuffer& to,
				 edm::ProductRegistry& prods):
    filename_(filename),
    ist_(filename_.c_str(),ios_base::binary | ios_base::in),
    reader_(ist_),
    to_(&to)
  {
    if(!ist_)
      {
	throw cms::Exception("Configuration","TestFileReader")
	  << "cannot open file " << filename_;
      }

    std::auto_ptr<SendJobHeader> p = readHeaderFromStream(ist_);
    // just get rid of the header
    if(edm::registryIsSubset(*p,prods)==false)
      {
	throw cms::Exception("Configuration","TestFileReader")
	  << "the header record in flie " << filename_
	  << "is not consistent with the one for the program \n";
      }
  }

  TestFileReader::~TestFileReader()
  {
  }

  void TestFileReader::start()
  {
    me_.reset(new boost::thread(boost::bind(TestFileReader::run,this)));
  }

  void TestFileReader::join()
  {
    me_->join();
  }

  void TestFileReader::run(TestFileReader* t)
  {
    t->readEvents();
  }

  void TestFileReader::readEvents()
  {
    while(1)
      {
	EventBuffer::ProducerBuffer b(*to_);

	int len=0;
	ist_.read((char*)&len,sizeof(int));

	if(!ist_ || len==0) break;

	// Pay attention here.
	// This is a bit of a mess.
	// Here we allocate an array (on the heap), fill it with data
	// from the file, then pass the bare pointer off onto the queue.
	// The ownership is to be picked up by the code that pulls it
	// off the queue.  The current implementation of the queue
	// is primitive - it knows nothing about the types of things
	// that are on the queue.

	BufHelper data(len);
	stor::FragEntry* msg = 
	  new (b.buffer()) stor::FragEntry(data.get(),data.get(),len);
	ist_.read((char*)msg->buffer_address_,len);
	b.commit(len);
	data.release();
      }
  }

}
