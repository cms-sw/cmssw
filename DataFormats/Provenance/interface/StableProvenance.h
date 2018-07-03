#ifndef DataFormats_Provenance_StableProvenance_h
#define DataFormats_Provenance_StableProvenance_h

/*----------------------------------------------------------------------

StableProvenance: The full description of a product, excluding the parentage.
The parentage can change from event to event.

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ReleaseVersion.h"

#include <memory>

#include <iosfwd>
/*
  StableProvenance

  definitions:
  Product: The EDProduct to which a provenance object is associated

  Creator: The EDProducer that made the product.
*/

namespace edm {
  class StableProvenance {
  public:
    StableProvenance();

    StableProvenance(std::shared_ptr<BranchDescription const> const& p, ProductID const& pid);

    BranchDescription const& branchDescription() const {return *branchDescription_;}
    std::shared_ptr<BranchDescription const> const& constBranchDescriptionPtr() const {return branchDescription_;}

    BranchID const& branchID() const {return branchDescription().branchID();}
    BranchID const& originalBranchID() const {return branchDescription().originalBranchID();}
    std::string const& branchName() const {return branchDescription().branchName();}
    std::string const& className() const {return branchDescription().className();}
    std::string const& moduleLabel() const {return branchDescription().moduleLabel();}
    std::string const& moduleName() const {return branchDescription().moduleName();}
    std::string const& processName() const {return branchDescription().processName();}
    std::string const& productInstanceName() const {return branchDescription().productInstanceName();}
    std::string const& friendlyClassName() const {return branchDescription().friendlyClassName();}
    ProcessHistory const& processHistory() const {return *processHistory_;}
    ProcessHistory const* processHistoryPtr() const {return processHistory_;}
    bool getProcessConfiguration(ProcessConfiguration& pc) const;
    ReleaseVersion releaseVersion() const;
    std::set<std::string> const& branchAliases() const {return branchDescription().branchAliases();}

    void write(std::ostream& os) const;

    void setProcessHistory(ProcessHistory const& ph) {processHistory_ = &ph;}

    ProductID const& productID() const {return productID_;}

    void setProductID(ProductID const& pid) {
      productID_ = pid;
    }

    void setBranchDescription(std::shared_ptr<BranchDescription const> const& p) {
      branchDescription_ = p;
    }

    void swap(StableProvenance&);

  private:
    std::shared_ptr<BranchDescription const> branchDescription_;
    ProductID productID_;
    ProcessHistory const* processHistory_; // We don't own this
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, StableProvenance const& p) {
    p.write(os);
    return os;
  }

  bool operator==(StableProvenance const& a, StableProvenance const& b);

}
#endif
