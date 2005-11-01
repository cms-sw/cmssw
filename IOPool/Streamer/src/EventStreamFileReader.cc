
#include "IOPool/Streamer/src/EventStreamFileReader.h"
#include "IOPool/Streamer/interface/BufferArea.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <algorithm>
#include <iterator>

using namespace std;
using namespace edm;

namespace edmtestp
{  
  // ----------------------------------

  EventStreamFileReader::EventStreamFileReader(edm::ParameterSet const& ps,
					       edm::InputSourceDescription const& desc):
    edm::InputSource(desc),
    filename_(ps.getParameter<string>("fileName")),
    ist_(filename_.c_str(),ios_base::binary | ios_base::in),
    reader_(ist_)
  {
    if(!ist_)
      {
	throw cms::Exception("Configuration","EventStreamFileReader")
	  << "cannot open file " << filename_;
      }

    std::auto_ptr<SendJobHeader> p = readHeaderFromStream(ist_);
    edm::mergeWithRegistry(*p,productRegistry());
  }

  EventStreamFileReader::~EventStreamFileReader()
  {
  }

  std::auto_ptr<edm::EventPrincipal> EventStreamFileReader::read()
  {
    return reader_.read(productRegistry());
  }

}
