/*----------------------------------------------------------------------
$Id: SecondaryInputSource.cc,v 1.2 2005/10/11 21:48:46 wmtan Exp $
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
    this->read_(idx, number, result);
  }
}
