#ifndef DataFormats_Provenance_Provenance_h
#define DataFormats_Provenance_Provenance_h

/*----------------------------------------------------------------------
  
Provenance: The full description of a product and how it came into
existence.

----------------------------------------------------------------------*/
#include <iosfwd>

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/EventEntryInfo.h"
#include "DataFormats/Provenance/interface/EntryDescription.h"
#include "DataFormats/Provenance/interface/RunLumiEntryInfo.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"
#include "boost/shared_ptr.hpp"

/*
  Provenance

  definitions:
  Product: The EDProduct to which a provenance object is associated

  Creator: The EDProducer that made the product.

  Parents: The EDProducts used as input by the creator.
*/

namespace edm {
  class Provenance {
  public:
    explicit Provenance(ConstBranchDescription const& p);
    explicit Provenance(BranchDescription const& p);
    Provenance(ConstBranchDescription const& p, boost::shared_ptr<EventEntryInfo> entryDesc);
    Provenance(BranchDescription const& p, boost::shared_ptr<EventEntryInfo> entryDesc);
    Provenance(ConstBranchDescription const& p, boost::shared_ptr<RunLumiEntryInfo> entryDesc);
    Provenance(BranchDescription const& p, boost::shared_ptr<RunLumiEntryInfo> entryDesc);

    ~Provenance() {}

    EntryDescription const& event() const {return entryDescription();}
    BranchDescription const& product() const {return branchDescription_.me();}

    BranchDescription const& branchDescription() const {return branchDescription_.me();}
    ConstBranchDescription const& constBranchDescription() const {return branchDescription_;}
    EventEntryInfo const* branchEntryInfoPtr() const {return branchEntryInfoPtr_.get();}
    boost::shared_ptr<EventEntryInfo> branchEntryInfoSharedPtr() const {return branchEntryInfoPtr_;}
    EventEntryInfo const& branchEntryInfo() const {return *branchEntryInfoPtr_;}
    EntryDescription const& entryDescription() const {return branchEntryInfo().entryDescription();}
    BranchID const& branchID() const {return product().branchID();}
    std::string const& branchName() const {return product().branchName();}
    std::string const& className() const {return product().className();}
    std::string const& moduleLabel() const {return product().moduleLabel();}
    std::string const& moduleName() const {return entryDescription().moduleName();}
    PassID const& passID() const {return entryDescription().passID();}
    std::string const& processName() const {return product().processName();}
    ProductID const& productID() const {return branchEntryInfo().productID();}
    ProductStatus const& productStatus() const {return branchEntryInfo().productStatus();}
    std::string const& productInstanceName() const {return product().productInstanceName();}
    std::string const& friendlyClassName() const {return product().friendlyClassName();}
    std::set<ParameterSetID> const& psetIDs() const {return product().psetIDs();}
    ParameterSetID const& psetID() const {return entryDescription().psetID();}
    ReleaseVersion const& releaseVersion() const {return entryDescription().releaseVersion();}
    std::set<std::string> const& branchAliases() const {return product().branchAliases();}
    ModuleDescriptionID const& moduleDescriptionID() const {return entryDescription().moduleDescriptionID();}
    ModuleDescription const& moduleDescription() const {return entryDescription().moduleDescription();}
    bool isPresent() const {return productstatus::present(productStatus());}

    std::vector<BranchID> const& parents() const {return entryDescription().parents();}

    void write(std::ostream& os) const;

    void setEventEntryInfo(boost::shared_ptr<EventEntryInfo> bei) const;

  private:
    ConstBranchDescription const branchDescription_;
    mutable boost::shared_ptr<EventEntryInfo> branchEntryInfoPtr_;
  };
  
  inline
  std::ostream&
  operator<<(std::ostream& os, Provenance const& p) {
    p.write(os);
    return os;
  }

  bool operator==(Provenance const& a, Provenance const& b);
}
#endif
