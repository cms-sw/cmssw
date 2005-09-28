/*----------------------------------------------------------------------
$Id: RandomAccessInputSource.cc,v 1.7 2005/07/30 23:47:52 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/RandomAccessInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm {

  RandomAccessInputSource::RandomAccessInputSource(InputSourceDescription const& desc) :
      InputSource(desc) {
  }

  RandomAccessInputSource::~RandomAccessInputSource() {}

  std::auto_ptr<EventPrincipal>
  RandomAccessInputSource::readEvent(EventID const& id) {
    // Do we need any error handling (e.g. exception translation)
    // here?
    std::auto_ptr<EventPrincipal> ep(this->read(id));
    if (ep.get()) {
	ep->addToProcessHistory(process_);
    }
    return ep;
  }

  void
  RandomAccessInputSource::skipEvents(int offset) {
    this->skip(offset);
  }
}
