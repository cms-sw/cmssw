/*----------------------------------------------------------------------
$Id: SecondaryInputSource.cc,v 1.1 2005/09/28 05:17:13 wmtan Exp $
----------------------------------------------------------------------*/
#include "FWCore/Framework/interface/SecondaryInputSource.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm {


  SecondaryInputSource::SecondaryInputSource() {}

  SecondaryInputSource::~SecondaryInputSource() {}

  void
  SecondaryInputSource::readMany(int idx, int number,
     std::vector<EventPrincipal*>& result) {
    // Do we need any error handling (e.g. exception translation)
    // here?
    this->read(idx, number, result);
  }
}
