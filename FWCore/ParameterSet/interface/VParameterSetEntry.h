#ifndef ParameterSet_VParameterSetEntry_h
#define ParameterSet_VParameterSetEntry_h

/** How ParameterSets are nested inside ParameterSets
    The main feature is that they're made persistent
    using a ParameterSetID, and only reconstituted as needed,
    when the value_ptr = 0;
  */

#include "FWCore/Utilities/interface/value_ptr.h"
#include "FWCore/ParameterSet/interface/ParameterSetEntry.h"

#include <string>
#include <vector>
#include <iosfwd>

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

    bool isTracked() const {return tracked;}

    /// returns the VPSet, reconstituting it from the
    /// Registry, if necessary
    std::vector<ParameterSet> const& vpset() const;
    ParameterSet & psetInVector(int i);

    std::vector<ParameterSet>::size_type size() const { return vpset().size(); }

    void registerPsetsAndUpdateIDs();

    friend std::ostream & operator<<(std::ostream & os, VParameterSetEntry const& vpsetEntry);

  private:

    bool tracked;
    mutable value_ptr<std::vector<ParameterSet> > theVPSet;
    mutable value_ptr<std::vector<ParameterSetID> > theIDs;
  };
}
#endif
