#ifndef DataFormats_Provenance_BranchDescription_h
#define DataFormats_Provenance_BranchDescription_h

/*----------------------------------------------------------------------

BranchDescription: The full description of a Branch.
This description also applies to every product instance on the branch.

----------------------------------------------------------------------*/
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationID.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

#include <iosfwd>
#include <map>
#include <set>
#include <string>

/*
  BranchDescription

  definitions:
  The event-independent description of an EDProduct.

*/

namespace edm {
  class WrapperInterfaceBase;
  class BranchDescription {
  public:
    static int const invalidSplitLevel = -1;
    static int const invalidBasketSize = 0;
    enum MatchMode { Strict = 0,
                     Permissive };

    BranchDescription();

    BranchDescription(BranchType const& branchType,
                      std::string const& moduleLabel,
                      std::string const& processName,
                      std::string const& className,
                      std::string const& friendlyClassName,
                      std::string const& productInstanceName,
                      std::string const& moduleName,
                      ParameterSetID const& parameterSetID,
                      TypeWithDict const& theTypeWithDict,
                      bool produced = true,
                      std::set<std::string> const& aliases = std::set<std::string>());

    BranchDescription(BranchDescription const& aliasForBranch,
                      std::string const& moduleLabelAlias,
                      std::string const& poruductInstanceAlias);

    ~BranchDescription() {}

    void init() {
      initBranchName();
      initFromDictionary();
    }

    void initBranchName();

    void initFromDictionary();

    void write(std::ostream& os) const;

    void merge(BranchDescription const& other);

    std::string const& moduleLabel() const {return moduleLabel_;}
    std::string const& processName() const {return processName_;}
    BranchID const& branchID() const {return branchID_;}
    BranchID const& aliasForBranchID() const {return aliasForBranchID_;}
    bool isAlias() const {return aliasForBranchID_.isValid() && produced();}
    BranchID const& originalBranchID() const {return aliasForBranchID_.isValid() ? aliasForBranchID_ : branchID_;}
    std::string const& fullClassName() const {return fullClassName_;}
    std::string const& className() const {return fullClassName();}
    std::string const& friendlyClassName() const {return friendlyClassName_;}
    std::string const& productInstanceName() const {return productInstanceName_;}
    bool produced() const {return transient_.produced_;}
    void setProduced(bool isProduced) {transient_.produced_ = isProduced;}
    bool present() const {return !transient_.dropped_;}
    bool dropped() const {return transient_.dropped_;}
    void setDropped(bool isDropped) {transient_.dropped_ = isDropped;}
    bool onDemand() const {return transient_.onDemand_;}
    void setOnDemand(bool isOnDemand) {transient_.onDemand_ = isOnDemand;}
    bool transient() const {return transient_.transient_;}
    void setTransient(bool isTransient) {transient_.transient_ = isTransient;}
    TypeWithDict const& wrappedType() const {return transient_.wrappedType_;}
    void setWrappedType(TypeWithDict const& type) {transient_.wrappedType_ = type;}
    TypeWithDict const& unwrappedType() const {return transient_.unwrappedType_;}
    void setUnwrappedType(TypeWithDict const& type) {transient_.unwrappedType_ = type;}
    TypeID wrappedTypeID() const {return TypeID(transient_.wrappedType_.typeInfo());}
    TypeID unwrappedTypeID() const {return TypeID(transient_.unwrappedType_.typeInfo());}
    int splitLevel() const {return transient_.splitLevel_;}
    void setSplitLevel(int level) {transient_.splitLevel_ = level;}
    int basketSize() const {return transient_.basketSize_;}
    void setBasketSize(int size) {transient_.basketSize_ = size;}

    ParameterSetID const& parameterSetID() const {return transient_.parameterSetID_;}
    std::string const& moduleName() const {return transient_.moduleName_;}

    std::map<ProcessConfigurationID, ParameterSetID> const& parameterSetIDs() const {
      return transient_.parameterSetIDs_;
    }
    void insertParameterSetID(std::pair<ProcessConfigurationID, ParameterSetID> const& entry) {
      transient_.parameterSetIDs_.insert(entry);
    }
    std::map<ProcessConfigurationID, std::string> const& moduleNames() const {
      return transient_.moduleNames_;
    }
    void insertModuleName(std::pair<ProcessConfigurationID, std::string> const& entry) {
      transient_.moduleNames_.insert(entry);
    }
    ParameterSetID const& psetID() const;
    bool isPsetIDUnique() const {return parameterSetIDs().size() == 1;}
    std::set<std::string> const& branchAliases() const {return branchAliases_;}
    void insertBranchAlias(std::string const& alias) {
      branchAliases_.insert(alias);
    }
    std::string const& branchName() const {return transient_.branchName_;}
    void clearBranchName() {transient_.branchName_.clear();}
    BranchType const& branchType() const {return branchType_;}
    std::string const& wrappedName() const {return transient_.wrappedName_;}
    void setWrappedName(std::string const& name) {transient_.wrappedName_ = name;}
    WrapperInterfaceBase*& wrapperInterfaceBase() {return transient_.wrapperInterfaceBase_;}

    WrapperInterfaceBase const* getInterface() const;
    void updateFriendlyClassName();

    void initializeTransients() {transient_.reset();}

    struct Transients {
      Transients();

      void reset();

      // The parameter set id of the producer.
      // This is set if and only if produced_ is true.
      ParameterSetID parameterSetID_;

      // The module name of the producer.
      // This is set if and only if produced_ is true.
      std::string moduleName_;

      // The branch name, which is currently derivable fron the other attributes.
      std::string branchName_;

      // The wrapped class name, which is currently derivable fron the other attributes.
      std::string wrappedName_;

      // Was this branch produced in this process rather than in a previous process
      bool produced_;

      // Was this branch produced in this current process and by unscheduled production
      // This item is set only in the framework, not by FWLite.
      bool onDemand_;

      // Has the branch been dropped from the product tree in this file
      // (or if this is a merged product registry, in the first file).
      // This item is set only in the framework, not by FWLite.
      bool dropped_;

      // ID's of process configurations for products on this branch
      //  with corresponding parameter set IDs,
      // This is initialized if and only if produced_ is false.
      std::map<ProcessConfigurationID, ParameterSetID> parameterSetIDs_;

      // ID's of process configurations for products on this branch
      //  with corresponding module names
      // This is initialized if and only if produced_ is false.
      std::map<ProcessConfigurationID, std::string> moduleNames_;

      // Is the class of the branch marked as transient
      // in the data dictionary
      bool transient_;

      // A TypeWithDict object for the wrapped object
      TypeWithDict wrappedType_;

      // A TypeWithDict object for the unwrapped object
      TypeWithDict unwrappedType_;

      // A pointer to a polymorphic object to obtain typed Wrapper.
      WrapperInterfaceBase* wrapperInterfaceBase_;

      // The split level of the branch, as marked
      // in the data dictionary.
      int splitLevel_;

      // The basket size of the branch, as marked
      // in the data dictionary.
      int basketSize_;
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
    BranchID branchID_;

    // the full name of the type of product this is
    std::string fullClassName_;

    // a readable name of the type of product this is
    std::string friendlyClassName_;

    // a user-supplied name to distinguish multiple products of the same type
    // that are produced by the same producer
    std::string productInstanceName_;

    // The branch ROOT alias(es), which are settable by the user.
    std::set<std::string> branchAliases_;

    // If this branch *is* an EDAlias, this field is the BranchID
    // of the branch for which this branch is an alias.
    // If this branch is not an EDAlias, the normal case, this field is 0.
    BranchID aliasForBranchID_;

    Transients transient_;
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
