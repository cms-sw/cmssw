#ifndef DataFormats_Provenance_BranchDescription_h
#define DataFormats_Provenance_BranchDescription_h

/*----------------------------------------------------------------------
  
BranchDescription: The full description of a Branch.
This description also applies to every product instance on the branch.  

----------------------------------------------------------------------*/
#include <iosfwd>
#include <string>
#include <map>
#include <set>

#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationID.h"
#include "DataFormats/Provenance/interface/Transient.h"

#include "Reflex/Type.h"
#include "FWCore/Utilities/interface/UseReflex.h"
/*
  BranchDescription

  definitions:
  The event-independent description of an EDProduct.

*/

namespace edm {
  class BranchDescription {
  public:
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

    ~BranchDescription() {}

    void init() const;

    void write(std::ostream& os) const;

    void merge(BranchDescription const& other);

    std::string const& moduleLabel() const {return moduleLabel_;}
    std::string const& processName() const {return processName_;}
    BranchID const& branchID() const {return branchID_;}
    ProductID const& oldProductID() const {return productID_;}
    std::string const& fullClassName() const {return fullClassName_;}
    std::string const& className() const {return fullClassName();}
    std::string const& friendlyClassName() const {return friendlyClassName_;}
    std::string const& productInstanceName() const {return productInstanceName_;} 
    bool & produced() const {return transients_.get().produced_;}
    bool present() const {return !transients_.get().dropped_;}
    bool & dropped() const {return transients_.get().dropped_;}
    bool & transient() const {return transients_.get().transient_;}
    Reflex::Type & type() const {return transients_.get().type_;}
    int & splitLevel() const {return transients_.get().splitLevel_;}
    int & basketSize() const {return transients_.get().basketSize_;}

    ParameterSetID const& parameterSetID() const {return transients_.get().parameterSetID_;}
    std::string const& moduleName() const {return transients_.get().moduleName_;}

    std::map<ProcessConfigurationID, ParameterSetID>& parameterSetIDs() const {
      return transients_.get().parameterSetIDs_;
    }
    std::map<ProcessConfigurationID, std::string>& moduleNames() const {
      return transients_.get().moduleNames_;
    }
    ParameterSetID const& psetID() const;
    bool isPsetIDUnique() const {return parameterSetIDs().size() == 1;}
    std::set<std::string> const& branchAliases() const {return branchAliases_;}
    std::set<std::string> & branchAliases() {return branchAliases_;}
    std::string & branchName() const {return transients_.get().branchName_;}
    BranchType const& branchType() const {return branchType_;}
    std::string & wrappedName() const {return transients_.get().wrappedName_;}

    void setDropped() const {dropped() = true;}
    void updateFriendlyClassName();

    struct Transients {
      Transients();

      // The parameter set id of the producer.
      // This is initialized if and only if produced_ is true.
      ParameterSetID parameterSetID_;

      // The module name of the producer.
      // This is initialized if and only if produced_ is true.
      std::string moduleName_;

      // The branch name, which is currently derivable fron the other attributes.
      std::string branchName_;

      // The wrapped class name, which is currently derivable fron the other attributes.
      std::string wrappedName_;

      // Was this branch produced in this process
      // rather than in a previous process
      bool produced_;

      // Has the branch been dropped from the product tree in this file
      // (or if this is a merged product registry, in the first file).
      bool dropped_;

      // Is the class of the branch marked as transient
      // in the data dictionary
      bool transient_;

      // The Reflex Type of the wrapped object.
      Reflex::Type type_;

      // The split level of the branch, as marked
      // in the data dictionary.
      int splitLevel_;

      // The basket size of the branch, as marked
      // in the data dictionary.
      int basketSize_;

      // ID's of process configurations for products on this branch
      //  with corresponding parameter set IDs,
      // This is initialized if and only if produced_ is false.
      mutable std::map<ProcessConfigurationID, ParameterSetID> parameterSetIDs_;

      // ID's of process configurations for products on this branch
      //  with corresponding module names
      // This is initialized if and only if produced_ is false.
      mutable std::map<ProcessConfigurationID, std::string> moduleNames_;
    };

  private:
    void throwIfInvalid_() const;

    // What tree is the branch in?
    BranchType branchType_;

    // A human friendly string that uniquely identifies the EDProducer
    // and becomes part of the identity of a product that it produces
    std::string moduleLabel_;

    // the physical process that this program was part of (e.g. production)
    std::string processName_;

    // An ID uniquely identifying the branch
    mutable BranchID branchID_;

    // An ID uniquely identifying the branch
    // This field is obsolete and is needed only for backward compatibility
    // with file format 7 and earlier.  
    ProductID productID_;

    // the full name of the type of product this is
    std::string fullClassName_;

    // a readable name of the type of product this is
    std::string friendlyClassName_;

    // a user-supplied name to distinguish multiple products of the same type
    // that are produced by the same producer
    std::string productInstanceName_;

    // The branch ROOT alias(es), which are settable by the user.
    std::set<std::string> branchAliases_;

    mutable Transient<Transients> transients_;
  };
  
  inline
  std::ostream&
  operator<<(std::ostream& os, BranchDescription const& p) {
    p.write(os);
    return os;
  }

  bool operator<(BranchDescription const& a, BranchDescription const& b);

  bool operator==(BranchDescription const& a, BranchDescription const& b);

  bool combinable(BranchDescription const& a, BranchDescription const& b);

  std::string match(BranchDescription const& a,
	BranchDescription const& b,
	std::string const& fileName,
	BranchDescription::MatchMode m);
}
#endif
