#ifndef DataFormats_Provenance_ConstBranchDescription_h
#define DataFormats_Provenance_ConstBranchDescription_h

/*----------------------------------------------------------------------

ConstBranchDescription: A class containing a constant shareable branch description
that is inexpensive to copy.
This class is not persistable.

----------------------------------------------------------------------*/
#include <iosfwd>
#include <string>
#include <map>
#include <set>

#include "boost/shared_ptr.hpp"

#include "DataFormats/Provenance/interface/BranchDescription.h"

/*
  ConstBranchDescription
*/

namespace edm {
  class ConstBranchDescription {
  public:
    explicit ConstBranchDescription(BranchDescription const& bd) :
      ptr_(new BranchDescription(bd)) {}

    void init() const {ptr_->init();}

    void write(std::ostream& os) const {ptr_->write(os);}

    std::string const& moduleLabel() const {return ptr_->moduleLabel();}
    std::string const& moduleName() const {return ptr_->moduleName();}
    std::string const& processName() const {return ptr_->processName();}
    BranchID const& branchID() const {return ptr_->branchID();}
    bool isAlias() const {return ptr_->isAlias();}
    BranchID const& originalBranchID() const {return ptr_->originalBranchID();}
    std::string const& fullClassName() const {return ptr_->fullClassName();}
    std::string const& className() const {return ptr_->fullClassName();}
    std::string const& friendlyClassName() const {return ptr_->friendlyClassName();}
    std::string const& productInstanceName() const {return ptr_->productInstanceName();}
    bool produced() const {return ptr_->produced();}
    bool dropped() const {return ptr_->dropped();}
    bool onDemand() const {return ptr_->onDemand();}
    bool present() const {return ptr_->present();}
    bool transient() const {return ptr_->transient();}
    TypeWithDict const& wrappedType() const {return ptr_->wrappedType();}
    TypeWithDict const& unwrappedType() const {return ptr_->unwrappedType();}
    TypeID wrappedTypeID() const {return ptr_->wrappedTypeID();}
    TypeID unwrappedTypeID() const {return ptr_->unwrappedTypeID();}
    int splitLevel() const {return ptr_->splitLevel();}
    int basketSize() const {return ptr_->basketSize();}

    ParameterSetID const& parameterSetID() const {return ptr_->parameterSetID();}
    std::map<ProcessConfigurationID, ParameterSetID> const& parameterSetIDs() const {return ptr_->parameterSetIDs();}
    ParameterSetID const& psetID() const {return ptr_->psetID();}
    bool isPsetIDUnique() const {return ptr_->parameterSetIDs().size() == 1;}
    std::set<std::string> const& branchAliases() const {return ptr_->branchAliases();}
    std::string const& branchName() const {return ptr_->branchName();}
    BranchType const& branchType() const {return ptr_->branchType();}
    std::string const& wrappedName() const {return ptr_->wrappedName();}
    WrapperInterfaceBase const* getInterface() const {return ptr_->getInterface();}

    BranchDescription const& me() const {return *ptr_;}

  private:
    boost::shared_ptr<BranchDescription> ptr_;
  };

  inline
  std::ostream&
  operator<<(std::ostream& os, ConstBranchDescription const& p) {
    os << p.me();
    return os;
  }

  inline
  bool operator<(ConstBranchDescription const& a, ConstBranchDescription const& b) {
    return a.me() < b.me();
  }

  inline
  bool operator==(ConstBranchDescription const& a, ConstBranchDescription const& b) {
    return a.me() == b.me();
  }

  inline
  std::string match(ConstBranchDescription const& a,
                    ConstBranchDescription const& b,
                    std::string const& fileName,
                    BranchDescription::MatchMode m) {
    return match(a.me(), b.me(), fileName, m);
  }
}
#endif
