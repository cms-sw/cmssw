#ifndef DataFormats_Provenance_EventEntryDescription_h
#define DataFormats_Provenance_EventEntryDescription_h

/*----------------------------------------------------------------------
  
EventEntryDescription: The event dependent portion of the description of a product
and how it came into existence.

----------------------------------------------------------------------*/
#include <iosfwd>
#include <vector>
#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/EntryDescriptionID.h"

/*
  EventEntryDescription

  definitions:
  Product: The EDProduct to which a provenance object is associated

  Creator: The EDProducer that made the product.

  Parents: The EDProducts used as input by the creator.
*/

namespace edm {
  struct EventEntryDescription {
    EventEntryDescription();

    ~EventEntryDescription() {}

    // Only the 'salient attributes' are encoded into the ID.
    EntryDescriptionID id() const;

    // The Branch IDs of the parents
    std::vector<BranchID> parents_;

    // the last of these is not in the roadmap, but is on the board

    ModuleDescriptionID moduleDescriptionID_;

    // transient.  Filled in from the hash when needed.
    mutable boost::shared_ptr<ModuleDescription> moduleDescriptionPtr_; //! transient

    void init() const;

    void write(std::ostream& os) const;

    std::string const& moduleName() const {init(); return moduleDescriptionPtr_->moduleName_;}
    PassID const& passID() const {init(); return moduleDescriptionPtr_->passID();}
    ParameterSetID const& psetID() const {init(); return moduleDescriptionPtr_->parameterSetID();}
    ReleaseVersion const& releaseVersion() const {init(); return moduleDescriptionPtr_->releaseVersion();}
    std::vector<BranchID> const& parents() const {return parents_;}

    ModuleDescriptionID const& moduleDescriptionID() const {return moduleDescriptionID_;}
    ModuleDescription const& moduleDescription() const {init(); return *moduleDescriptionPtr_;}
  };
  
  inline
  std::ostream&
  operator<<(std::ostream& os, EventEntryDescription const& p) {
    p.write(os);
    return os;
  }

  // Only the 'salient attributes' are testing in equality comparison.
  bool operator==(EventEntryDescription const& a, EventEntryDescription const& b);
  inline bool operator!=(EventEntryDescription const& a, EventEntryDescription const& b) { return !(a==b); }
}
#endif
