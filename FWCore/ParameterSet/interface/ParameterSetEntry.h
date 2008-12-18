#ifndef ParameterSet_ParameterSetEntry_h
#define ParameterSet_ParameterSetEntry_h

/** How ParameterSets are nested inside ParameterSets
    The main feature is that they're made persistent
    using a ParameterSetID, and only reconstituted as needed,
    when the value_ptr = 0;
  */

#include "FWCore/Utilities/interface/value_ptr.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"

namespace edm {

  // forward declaration
  class ParameterSet;

  class ParameterSetEntry
  {
  public:
    // default ctor for serialization
    ParameterSetEntry();
    ParameterSetEntry(ParameterSet const& pset, bool isTracked);
    ParameterSetEntry(std::string const& rep);

    ~ParameterSetEntry();

    std::string toString() const;
    int sizeOfString() const;

    bool isTracked() const {return tracked;}

    ParameterSetID id() const {return theID;}
  
    /// returns the PSet, reconstituting it from the
    /// Registry, if necessary
    ParameterSet const& pset() const;
    ParameterSet & pset();

    /// we expect this to only be called by ParameterSet, on tracked psets
    void updateID() const;

    friend std::ostream & operator<<(std::ostream & os, ParameterSetEntry const& psetEntry);

  private:
    
    bool tracked;
    // can be internally reconstituted from the ID, in an
    // ostensibly const function
    mutable value_ptr<ParameterSet> thePSet;

    // mutable so save() can serialize it as late as possible
    mutable ParameterSetID theID;


  };

}

#endif

