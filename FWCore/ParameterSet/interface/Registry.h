#ifndef FWCore_ParameterSet_Registry_h
#define FWCore_ParameterSet_Registry_h

// ----------------------------------------------------------------------
// Declaration for pset::Registry. This is an implementation detail of
// the ParameterSet library.
//
// A Registry is used to keep track of the persistent form of all
// ParameterSets used a given program, so that they may be retrieved by
// ParameterSetID, and so they may be written to persistent storage.
// ----------------------------------------------------------------------

#include <map>

#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"



namespace edm {
  namespace pset {

    class ProcessParameterSetIDCache {
    public:
      ProcessParameterSetIDCache() : id_() { }
      ParameterSetID id() const { return id_; }
      void setID(ParameterSetID const& id) { id_ = id; }
    private:
      ParameterSetID id_;      
    };

    typedef detail::ThreadSafeRegistry<ParameterSetID,
    					ParameterSet,
					ProcessParameterSetIDCache>
                                        Registry;

    /// Associated free functions.

    /// Return the ParameterSetID of the top-level ParameterSet stored
    /// in the given Registry. Note the the returned ParameterSetID may
    /// be invalid; this will happen if the Registry has not yet been
    /// filled.
    ParameterSetID getProcessParameterSetID(Registry const* reg);

    /// Fill the given map with the persistent form of each
    /// ParameterSet in the given registry.
    typedef std::map<ParameterSetID, ParameterSetBlob> regmap_type;
    void fillMap(Registry* reg, regmap_type& fillme);

  }  // namespace pset

  ParameterSet getProcessParameterSet();

}  // namespace edm


#endif
