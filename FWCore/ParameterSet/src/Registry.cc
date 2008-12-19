// ----------------------------------------------------------------------
// $Id: Registry.cc,v 1.11 2008/12/18 05:14:04 wmtan Exp $
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
      // Note: The tracked part is in the registry.
      // The full parameter set including the untracked parts may also be there.
      for (iter i = reg->begin(), e = reg->end(); i != e; ++i) {
        // Note: The tracked part of the parameter set is in the registry.
        // The full parameter set including the untracked parts may also be there.
	// Persist only the former.
	std::string stringOfAll;
	i->second.toString(stringOfAll);
	std::string stringOfTracked;
	i->second.trackedPart().toString(stringOfTracked);
	if (stringOfTracked == stringOfAll) {
	  // This parameter set contains no untracked parts.
	  // Persist it.
	  fillme[i->first].pset_ = stringOfTracked;
	}
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

