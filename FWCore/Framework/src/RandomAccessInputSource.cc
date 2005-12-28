/*----------------------------------------------------------------------
$Id: RandomAccessInputSource.cc,v 1.2 2005/12/28 00:32:04 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/RandomAccessInputSource.h"

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
