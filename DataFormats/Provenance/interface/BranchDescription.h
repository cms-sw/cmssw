#ifndef DataFormats_Provenance_BranchDescription_h
#define DataFormats_Provenance_BranchDescription_h

/*----------------------------------------------------------------------
  
BranchDescription: The full description of a Branch.
This description also applies to every product instance on the branch.  

$Id: BranchDescription.h,v 1.5 2007/08/28 14:32:10 wmtan Exp $
----------------------------------------------------------------------*/
#include <iosfwd>
#include <string>
#include <set>

#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ModuleDescriptionID.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationID.h"

#include "Reflex/Type.h"
/*
  BranchDescription

  definitions:
  The event-independent description of an EDProduct.

*/

namespace edm {
  struct BranchDescription {
    static int const invalidSplitLevel = -1;
    static int const invalidBasketSize = 0;
    enum MatchMode { Strict = 0,
		     Permissive };

    BranchDescription();

    BranchDescription(BranchType const& branchType,
		      std::string const& mdLabel, 
		      std::string const& procName, 
		      std::string const& name, 
		      std::string const& fName, 
		      std::string const& pin, 
		      ModuleDescription const& modDesc,
		      std::set<std::string> const& aliases = std::set<std::string>());


    BranchDescription(BranchType const& branchType,
		      std::string const& mdLabel, 
		      std::string const& procName, 
		      std::string const& name, 
		      std::string const& fName, 
		      std::string const& pin, 
		      ModuleDescriptionID const& mdID, // = ModuleDescriptionID(),
		      std::set<ParameterSetID> const& psIDs, // = std::set<ParameterSetID>(),
		      std::set<ProcessConfigurationID> const& procConfigIDs, // = std::set<ProcessConfigurationID>(),
		      std::set<std::string> const& aliases = std::set<std::string>());

    ~BranchDescription() {}

    void init() const;

    void write(std::ostream& os) const;

    void merge(BranchDescription const& other);

    std::string const& moduleLabel() const {return moduleLabel_;}
    std::string const& processName() const {return processName_;}
    ProductID const& productID() const {return productID_;}
    std::string const& fullClassName() const {return fullClassName_;}
    std::string const& className() const {return fullClassName();}
    std::string const& friendlyClassName() const {return friendlyClassName_;}
    std::string const& productInstanceName() const {return productInstanceName_;} 
    bool const& produced() const {return produced_;}
    bool const& present() const {return present_;}
    bool const& provenancePresent() const {return provenancePresent_;}
    bool const& transient() const {return transient_;}
    ROOT::Reflex::Type const& type() const {return type_;}
    int const& splitLevel() const {return splitLevel_;}
    int const& basketSize() const {return basketSize_;}

    ModuleDescriptionID const& moduleDescriptionID() const {return moduleDescriptionID_;}
    std::set<ParameterSetID> const& psetIDs() const {return psetIDs_;}
    ParameterSetID const& psetID() const;
    bool isPsetIDUnique() const {return psetIDs().size() == 1;}
    std::set<ProcessConfigurationID> const& processConfigurationIDs() const {return processConfigurationIDs_;}
    std::set<std::string> const& branchAliases() const {return branchAliases_;}
    std::string const& branchName() const {return branchName_;}
    BranchType const& branchType() const {return branchType_;}

  private:
    void throwIfInvalid_() const;

  //TODO: Make all the data private!
  public:

    // What tree is the branch in?
    BranchType branchType_;

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
    // This is just used as a cache, and is not logically
    // part of the branch description.
    mutable ModuleDescriptionID moduleDescriptionID_;

    // ID's of parameter set of the creators of products
    // on this branch
    std::set<ParameterSetID> psetIDs_;

    // ID's of process configurations for products
    // on this branch
    std::set<ProcessConfigurationID> processConfigurationIDs_;

    // The branch ROOT alias(es), which are settable by the user.
    std::set<std::string> branchAliases_;

    // The branch name (transient), which is currently derivable fron the other
    // attributes.
    mutable std::string branchName_;

    // Was this branch produced in this process
    // rather than in a previous process
    bool produced_;

    // Is the branch present in the product tree
    // in the input file (or any of the input files)
    mutable bool present_;

    // Is the branch present in the provenance tree
    // in the input file (or any of the input files)
    mutable bool provenancePresent_;

    // Is the class of the branch marked as transient
    // in the data dictionary
    mutable bool transient_;

    // The Reflex Type of the wrapped object.
    mutable ROOT::Reflex::Type type_;

    // The split level of the branch, as marked
    // in the data dictionary.
    mutable int splitLevel_;

    // The basket size of the branch, as marked
    // in the data dictionary.
    mutable int basketSize_;
  };
  
  inline
  std::ostream&
  operator<<(std::ostream& os, BranchDescription const& p) {
    p.write(os);
    return os;
  }

  bool operator<(BranchDescription const& a, BranchDescription const& b);

  bool operator==(BranchDescription const& a, BranchDescription const& b);

  std::string match(BranchDescription const& a,
	BranchDescription const& b,
	std::string const& fileName,
	BranchDescription::MatchMode m);
}
#endif
