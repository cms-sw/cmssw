#ifndef ParameterSet_ParameterSetEntry_h
#define ParameterSet_ParameterSetEntry_h

/** How ParameterSets are nested inside ParameterSets
    The main feature is that they're made persistent
    using a ParameterSetID, and only reconstituted as needed,
    when the value_ptr = 0;
  */

#include "FWCore/Utilities/interface/atomic_value_ptr.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"

namespace cms {
  class Digest;
}

namespace edm {

  // forward declaration
  class ParameterSet;

  class ParameterSetEntry {
  public:
    // default ctor for serialization
    ParameterSetEntry();
    ParameterSetEntry(ParameterSet const& pset, bool isTracked);
    ParameterSetEntry(ParameterSetID const& id, bool isTracked);
    explicit ParameterSetEntry(std::string const& rep);

    ~ParameterSetEntry();

    std::string toString() const;
    void toString(std::string& result) const;
    void toDigest(cms::Digest &digest) const;

    bool isTracked() const {return isTracked_;}
    void setIsTracked(bool v) { isTracked_ = v; }

    ParameterSetID id() const {return theID_;}
  
    /// returns the PSet
    ParameterSet const& pset() const;
    ParameterSet& psetForUpdate();
    /// reconstitutes the PSet from the registry
    void fillPSet() const;

    void updateID();

    std::string dump(unsigned int indent = 0) const;
    friend std::ostream & operator<<(std::ostream & os, ParameterSetEntry const& psetEntry);

  private:
    
    bool isTracked_;
    // can be internally reconstituted from the ID, in an
    // ostensibly const function
    mutable atomic_value_ptr<ParameterSet> thePSet_;

    ParameterSetID theID_;
  };

}

#endif

