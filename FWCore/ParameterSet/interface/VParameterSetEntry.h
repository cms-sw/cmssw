#ifndef ParameterSet_VParameterSetEntry_h
#define ParameterSet_VParameterSetEntry_h

/** How ParameterSets are nested inside ParameterSets
    The main feature is that they're made persistent
    using a ParameterSetID, and only reconstituted as needed,
    when the value_ptr = 0;
  */

#include <vector>
#include "FWCore/Utilities/interface/value_ptr.h"
#include "FWCore/ParameterSet/interface/ParameterSetEntry.h"

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
    int sizeOfString() const;

    bool isTracked() const {return tracked;}

    /// returns the VPSet, reconstituting it from the
    /// Registry, if necessary
    std::vector<ParameterSet> const& vpset() const;
    std::vector<ParameterSet>& vpset();

    std::vector<ParameterSetEntry> const& psetEntries() const {return thePSetEntries;}

    void updateIDs() const;

    friend std::ostream & operator<<(std::ostream & os, VParameterSetEntry const& vpsetEntry);

  private:

    bool tracked;
    mutable value_ptr<std::vector<ParameterSet> > theVPSet;
    mutable std::vector<ParameterSetEntry> thePSetEntries;
  };

}

#endif

