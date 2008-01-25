#ifndef DataFormats_Provenance_EntryDescription_h
#define DataFormats_Provenance_EntryDescription_h

/*----------------------------------------------------------------------
  
EntryDescription: The event dependent portion of the description of a product
and how it came into existence.

$Id: EntryDescription.h,v 1.1 2007/03/04 04:48:08 wmtan Exp $
----------------------------------------------------------------------*/
#include <iosfwd>
#include <vector>
#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"

/*
  EntryDescription

  definitions:
  Product: The EDProduct to which a provenance object is associated

  Creator: The EDProducer that made the product.

  Parents: The EDProducts used as input by the creator.
*/

namespace edm {
  struct EntryDescription {
    EntryDescription();
    EntryDescription(ProductID const& pid);

    ~EntryDescription() {}

    ProductID productID_;

    // The EDProduct IDs of the parents
    std::vector<ProductID> parents_;

    void mergeEntryDescription(EntryDescription const* entry);

    // the last of these is not in the roadmap, but is on the board

    ModuleDescriptionID moduleDescriptionID_;

    // transient.  Filled in from the hash when needed.
    mutable boost::shared_ptr<ModuleDescription> moduleDescriptionPtr_;

    void init() const;

    void write(std::ostream& os) const;

    std::string const& moduleName() const {init(); return moduleDescriptionPtr_->moduleName_;}
    PassID const& passID() const {init(); return moduleDescriptionPtr_->passID();}
    ParameterSetID const& psetID() const {init(); return moduleDescriptionPtr_->parameterSetID();}
    ReleaseVersion const& releaseVersion() const {init(); return moduleDescriptionPtr_->releaseVersion();}
    std::vector<ProductID> const& parents() const {return parents_;}

    ModuleDescriptionID const& moduleDescriptionID() const {return moduleDescriptionID_;}
    ModuleDescription const& moduleDescription() const {init(); return *moduleDescriptionPtr_;}
  };
  
  inline
  std::ostream&
  operator<<(std::ostream& os, EntryDescription const& p) {
    p.write(os);
    return os;
  }

  bool operator==(EntryDescription const& a, EntryDescription const& b);
}
#endif
