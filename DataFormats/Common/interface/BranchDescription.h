#ifndef Common_BranchDescription_h
#define Common_BranchDescription_h

/*----------------------------------------------------------------------
  
BranchDescription: The full description of a Branch.
This description also applies to every product instance on the branch.  

$Id: BranchDescription.h,v 1.11.2.6 2006/07/01 06:21:57 wmtan Exp $
----------------------------------------------------------------------*/
#include <ostream>
#include <string>
#include <set>

#include "DataFormats/Common/interface/ProductID.h"
#include "DataFormats/Common/interface/ParameterSetID.h"
#include "DataFormats/Common/interface/ModuleDescriptionID.h"

/*
  BranchDescription

  definitions:
  The event-independent description of an EDProduct.

*/

namespace edm {
  class EDProduct;
  struct BranchDescription {

    BranchDescription();

    explicit BranchDescription(std::string const& moduleLabel, 
		std::string const& processName, 
		std::string const& name, 
		std::string const& fName, 
		std::string const& pin, 
		ModuleDescriptionID const& mdID = ModuleDescriptionID(),
		std::set<ParameterSetID> const& psetIDs = std::set<ParameterSetID>(),
		std::set<std::string> const& aliases = std::set<std::string>());

    ~BranchDescription() {}

    // A human friendly string that uniquely identifies the EDProducer
    // and becomes part of the identity of a product that it produces
    std::string moduleLabel_;

    // the physical process that this program was part of (e.g. production)
    std::string processName_;

    // An ID uniquely identifying the branch
    ProductID productID_;

    // the full name of the type of product this is
    std::string fullClassName_;

    // a readable name of the type of product this is
    std::string friendlyClassName_;

    // a user-supplied name to distinguish multiple products of the same type
    // that are produced by the same producer
    std::string productInstanceName_;

    // The module description id of the producer (transient).
    // This is only valid if produced_ is true.
    mutable ModuleDescriptionID moduleDescriptionID_;

    // ID's of parameter set of the creators of products
    // on this branch
    std::set<ParameterSetID> psetIDs_;

    // The branch ROOT alias(es), which arw settable by the user.
    mutable std::set<std::string> branchAliases_;

    // The branch name (transient), which is currently derivable fron the other
    // attributes.
    mutable std::string branchName_;

    // Was this branch produced in this process
    // rather than in a previous process
    bool produced_;

    void init() const;

    void write(std::ostream& os) const;

    std::string const& branchName() const {return branchName_;}
    std::string const& className() const {return fullClassName_;}
    std::string const& moduleLabel() const {return moduleLabel_;}
    std::string const& processName() const {return processName_;}
    ProductID const& productID() const {return productID_;}
    std::string const& productInstanceName() const {return productInstanceName_;} 
    std::string const& friendlyClassName() const {return friendlyClassName_;}
    std::set<std::string> const& branchAliases() const {return branchAliases_;}
    std::set<ParameterSetID> const& psetIDs() const {return psetIDs_;}
  };
  
  inline
  std::ostream&
  operator<<(std::ostream& os, const BranchDescription& p) {
    p.write(os);
    return os;
  }

  bool operator<(BranchDescription const& a, BranchDescription const& b);

  bool operator==(BranchDescription const& a, BranchDescription const& b);

}
#endif
