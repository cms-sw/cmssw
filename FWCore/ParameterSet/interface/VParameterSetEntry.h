#ifndef FWCore_ParameterSet_VParameterSetEntry_h
#define FWCore_ParameterSet_VParameterSetEntry_h

/** How ParameterSets are nested inside ParameterSets
    The main feature is that they're made persistent
    using a ParameterSetID, and only reconstituted as needed,
    when the value_ptr = 0;
  */

#include "FWCore/ParameterSet/interface/ParameterSetEntry.h"
#include "FWCore/Utilities/interface/value_ptr.h"
#include "FWCore/Utilities/interface/atomic_value_ptr.h"

#include <iosfwd>
#include <string>
#include <vector>

namespace edm {

  // forward declaration
  class ParameterSet;

  class VParameterSetEntry {
  public:
    // default ctor for serialization
    VParameterSetEntry();
    VParameterSetEntry(std::vector<ParameterSet> const& vpset, bool isTracked);
    VParameterSetEntry(std::string const& rep);

    ~VParameterSetEntry();

    std::string toString() const;
    void toString(std::string& result) const;
    void toDigest(cms::Digest &digest) const;

    bool isTracked() const {return tracked_;}

    /// returns the VPSet
    std::vector<ParameterSet> const& vpset() const;
    std::vector<ParameterSet>& vpsetForUpdate();
    /// reconstitutes the VPSet from the registry
    void fillVPSet() const;
    ParameterSet& psetInVector(int i);

    std::vector<ParameterSet>::size_type size() const { return vpset().size(); }

    void registerPsetsAndUpdateIDs();

    std::string dump(unsigned int indent = 0) const;
    friend std::ostream& operator<<(std::ostream& os, VParameterSetEntry const& vpsetEntry);

  private:

    bool tracked_;
    mutable atomic_value_ptr<std::vector<ParameterSet> > theVPSet_;
    value_ptr<std::vector<ParameterSetID> > theIDs_;
  };
}
#endif
