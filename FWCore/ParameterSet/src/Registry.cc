// ----------------------------------------------------------------------
// $Id: Registry.cc,v 1.10.4.3 2008/12/16 08:47:45 wmtan Exp $
//
// ----------------------------------------------------------------------

#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/EDMException.h"


namespace edm {
  namespace pset {
    ParameterSetID
    getProcessParameterSetID(Registry const* reg) {
      return reg->extra().id();
    }

    void fillMap(Registry* reg, regmap_type& fillme) {
      typedef Registry::const_iterator iter;
      fillme.clear();
      for (iter i=reg->begin(), e=reg->end(); i != e; ++i) {
	fillme[i->first].pset_ = i->second.toStringOfTracked();
      }
    }
  } // namespace pset

  ParameterSet getProcessParameterSet() {
    pset::Registry* reg = pset::Registry::instance();
    ParameterSetID id = pset::getProcessParameterSetID(reg);

    ParameterSet result;
    if (!reg->getMapped(id, result))
      throw edm::Exception(errors::EventCorruption, "Unknown ParameterSetID")
	<< "Unable to find the ParameterSet for id: "
	<< id
	<< ";\nthis was supposed to be the process ParameterSet\n";

    return result;
  }

} // namespace edm

