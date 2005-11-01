
#include "IOPool/Streamer/interface/TestFileReader.h"
#include "IOPool/Streamer/interface/BufferArea.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "boost/bind.hpp"

#include <algorithm>
#include <iterator>

using namespace std;
using namespace edm;

namespace edmtestp
{  
  // ----------------------------------

  TestFileReader::TestFileReader(const std::string& filename,
				 edm::EventBuffer& to,
				 edm::ProductRegistry& prods):
    filename_(filename),
    ist_(filename_.c_str(),ios_base::binary | ios_base::in),
    reader_(ist_),
    inserter_(to),
    to_(&to),
    prods_(&prods)
  {
    if(!ist_)
      {
	throw cms::Exception("Configuration","TestFileReader")
	  << "cannot open file " << filename_;
      }

    std::auto_ptr<SendJobHeader> p = readHeaderFromStream(ist_);
    // just get rid of the header
    if(edm::registryIsSubset(*p,*prods_)==false)
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
    EventReader::Buf b(1000*1000*7);
    int len;

    while(1)
      {
	// done if length equals 0
	if((len=reader_.readMessage(b))==0) break;
	edm::EventMsg msg(&b[0],len);
	inserter_.insert(msg,*prods_);
      }
    
  }

}
