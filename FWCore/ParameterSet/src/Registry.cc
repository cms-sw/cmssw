// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/Utilities/interface/EDMException.h"


namespace edm {
  namespace pset {
    ParameterSetID
    getProcessParameterSetID(Registry const* reg) {
      ParameterSetID const& psetID = reg->extra().id();
      if (!psetID.isValid()) {
        throw edm::Exception(errors::LogicError)
          << "Illegal attempt to access the process top level parameter set ID\n"
          << "before that parameter set has been frozen and registered.\n"
          << "The parameter set can be changed during module validation,\n"
	  << "which occurs concurrently with module construction.\n"
          << "It is illegal to access the parameter set before it is frozen.\n";
      }
      return psetID;
    }

    void fillMap(Registry* reg, regmap_type& fillme) {
      typedef Registry::const_iterator iter;
      fillme.clear();
      // Note: The tracked part is in the registry.
      for (iter i = reg->begin(), e = reg->end(); i != e; ++i) {
	fillme[i->first].pset() = i->second.toString();
      }
    }
  } // namespace pset

  ParameterSet const& getProcessParameterSet() {
    pset::Registry* reg = pset::Registry::instance();
    ParameterSetID id = pset::getProcessParameterSetID(reg);

    ParameterSet const* result;
    if (0==(result=reg->getMapped(id))) {
      throw edm::Exception(errors::EventCorruption, "Unknown ParameterSetID")
	<< "Unable to find the ParameterSet for id: "
	<< id
	<< ";\nthis was supposed to be the process ParameterSet\n";
    }
    return *result;
  }

} // namespace edm

#include "FWCore/Utilities/interface/ThreadSafeRegistry.icc"
DEFINE_THREAD_SAFE_REGISTRY_INSTANCE(edm::pset::Registry)
