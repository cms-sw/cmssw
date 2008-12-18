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
#include "DataFormats/Provenance/interface/ModuleDescriptionID.h"
#include "DataFormats/Provenance/interface/EntryDescriptionID.h"
#include "DataFormats/Provenance/interface/Transient.h"

/*
  EventEntryDescription

  definitions:
  Product: The EDProduct to which a provenance object is associated

  Creator: The EDProducer that made the product.

  Parents: The EDProducts used as input by the creator.
*/

namespace edm {
  class EventEntryDescription {
  public:
    EventEntryDescription();

    ~EventEntryDescription() {}

    // Only the 'salient attributes' are encoded into the ID.
    EntryDescriptionID id() const;

    void write(std::ostream& os) const;

    std::string const& moduleName() const {return getModuleDescriptionPtr()->moduleName_;}
    PassID const& passID() const {return getModuleDescriptionPtr()->passID();}
    ParameterSetID const& psetID() const {return getModuleDescriptionPtr()->parameterSetID();}
    ReleaseVersion const& releaseVersion() const {return getModuleDescriptionPtr()->releaseVersion();}
    std::vector<BranchID> const& parents() const {return parents_;}
    std::vector<BranchID> & parents() {return parents_;}

    ModuleDescriptionID const& moduleDescriptionID() const {return moduleDescriptionID_;}
    ModuleDescriptionID & moduleDescriptionID() {return moduleDescriptionID_;}
    ModuleDescription const& moduleDescription() const {return *getModuleDescriptionPtr();}

    struct Transients {
      Transients() : moduleDescriptionPtr_() {}
      boost::shared_ptr<ModuleDescription> moduleDescriptionPtr_;
    };

  private:
    void init() const;

    boost::shared_ptr<ModuleDescription> & getModuleDescriptionPtr() const {
      init();
      return transients_.get().moduleDescriptionPtr_;
    }

    boost::shared_ptr<ModuleDescription> & moduleDescriptionPtr() const {
      return transients_.get().moduleDescriptionPtr_;
    }

    // The Branch IDs of the parents
    std::vector<BranchID> parents_;

    // the last of these is not in the roadmap, but is on the board

    ModuleDescriptionID moduleDescriptionID_;

    mutable Transient<Transients> transients_;
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
