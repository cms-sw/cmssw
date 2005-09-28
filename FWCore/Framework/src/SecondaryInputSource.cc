/*----------------------------------------------------------------------
$Id: SecondaryInputSource.cc,v 1.7 2005/07/30 23:47:52 wmtan Exp $
----------------------------------------------------------------------*/
#include "FWCore/Framework/interface/SecondaryInputSource.h"
#include "FWCore/Framework/interface/InputSourceDescription.h"
#include "FWCore/Framework/interface/EventPrincipal.h"

namespace edm {

  SecondaryInputSource::SecondaryInputSource(InputSourceDescription const& desc) : InputSource(desc) {
  }

  SecondaryInputSource::SecondaryInputSource() : InputSource(InputSourceDescription()) {}

  SecondaryInputSource::~SecondaryInputSource() {}

  std::auto_ptr<EventPrincipal>
  SecondaryInputSource:: read() {
    // part of KLUDGE
    return std::auto_ptr<EventPrincipal>(0);
  }

  void
  SecondaryInputSource::readMany(int idx, int number,
     std::vector<EventPrincipal*>& result) {
    // Do we need any error handling (e.g. exception translation)
    // here?
    this->read(idx, number, result);
  }
}
