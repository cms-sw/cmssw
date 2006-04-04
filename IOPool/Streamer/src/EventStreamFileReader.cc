
#include "IOPool/Streamer/src/EventStreamFileReader.h"
#include "IOPool/Streamer/interface/BufferArea.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "IOPool/Streamer/interface/ClassFiller.h"
#include "IOPool/Streamer/interface/Utilities.h"

#include <algorithm>
#include <iterator>

using namespace std;
using namespace edm;

namespace edmtestp
{  
  // ----------------------------------

  EventStreamFileReader::EventStreamFileReader(edm::ParameterSet const& pset,
					       edm::InputSourceDescription const& desc):
    edm::InputSource(pset, desc),
    filename_(pset.getParameter<string>("fileName")),
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

    // jbk - the next line should not be needed
    edm::declareStreamers(productRegistry());
    edm::buildClassCache(productRegistry());
    loadExtraClasses();
  }

  EventStreamFileReader::~EventStreamFileReader()
  {
  }

  std::auto_ptr<edm::EventPrincipal> EventStreamFileReader::read()
  {
    return reader_.read(productRegistry());
  }

}
