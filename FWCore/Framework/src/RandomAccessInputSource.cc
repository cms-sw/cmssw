/*----------------------------------------------------------------------
$Id: RandomAccessInputSource.cc,v 1.1 2005/09/28 05:15:59 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/RandomAccessInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm {

  RandomAccessInputSource::RandomAccessInputSource(InputSourceDescription const& desc) :
      InputSource(desc) {
  }

  RandomAccessInputSource::~RandomAccessInputSource() {}

  void
  RandomAccessInputSource::skipEvents(int offset) {
    this->skip(offset);
  }
}
