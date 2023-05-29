/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProductRegistryHelper.h"
#include "DataFormats/Common/interface/setIsMergeable.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Reflection/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Reflection/interface/TypeWithDict.h"

#include <vector>
#include <typeindex>

namespace edm {
  ProductRegistryHelper::~ProductRegistryHelper() noexcept(false) {}

  ProductRegistryHelper::TypeLabelList const& ProductRegistryHelper::typeLabelList() const { return typeLabelList_; }

  namespace {
    void throwProducesWithoutAbility(const char* transitionName, std::string const& productTypeName) {
      throw edm::Exception(edm::errors::LogicError)
          << "Module declares it can produce a product of type \'" << productTypeName << "\'\nin a " << transitionName
          << ", but does not have the ability to produce in " << transitionName << "s.\n"
          << "You must add a template parameter of type " << transitionName << "Producer\n"
          << "or " << transitionName << "Producer to the EDProducer or EDFilter base class\n"
          << "of the module. Or you could remove the call to the function \'produces\'\n"
          << "(Note legacy modules are not ever allowed to produce in Runs or Lumis)\n";
    }
  }  // namespace

  void ProductRegistryHelper::addToRegistry(TypeLabelList::const_iterator const& iBegin,
                                            TypeLabelList::const_iterator const& iEnd,
                                            ModuleDescription const& iDesc,
                                            ProductRegistry& iReg,
                                            ProductRegistryHelper* iProd,
                                            bool iIsListener) {
    std::vector<std::string> missingDictionaries;
    std::vector<std::string> producedTypes;
    std::set<std::tuple<BranchType, std::type_index, std::string>> registeredProducts;

    for (TypeLabelList::const_iterator p = iBegin; p != iEnd; ++p) {
      if (p->transition_ == Transition::BeginRun && not iProd->hasAbilityToProduceInBeginRuns()) {
        throwProducesWithoutAbility("BeginRun", p->typeID_.userClassName());
      } else if (p->transition_ == Transition::EndRun && not iProd->hasAbilityToProduceInEndRuns()) {
        throwProducesWithoutAbility("EndRun", p->typeID_.userClassName());
      } else if (p->transition_ == Transition::BeginLuminosityBlock && not iProd->hasAbilityToProduceInBeginLumis()) {
        throwProducesWithoutAbility("BeginLuminosityBlock", p->typeID_.userClassName());
      } else if (p->transition_ == Transition::EndLuminosityBlock && not iProd->hasAbilityToProduceInEndLumis()) {
        throwProducesWithoutAbility("EndLuminosityBlock", p->typeID_.userClassName());
      } else if (p->transition_ == Transition::BeginProcessBlock &&
                 not iProd->hasAbilityToProduceInBeginProcessBlocks()) {
        throwProducesWithoutAbility("BeginProcessBlock", p->typeID_.userClassName());
      } else if (p->transition_ == Transition::EndProcessBlock && not iProd->hasAbilityToProduceInEndProcessBlocks()) {
        throwProducesWithoutAbility("EndProcessBlock", p->typeID_.userClassName());
      }
      if (!checkDictionary(missingDictionaries, p->typeID_)) {
        checkDictionaryOfWrappedType(missingDictionaries, p->typeID_);
        producedTypes.emplace_back(p->typeID_.className());
        continue;
      }
      auto branchType = convertToBranchType(p->transition_);
      if (branchType != InEvent) {
        std::tuple<BranchType, std::type_index, std::string> entry{
            branchType, p->typeID_.typeInfo(), p->productInstanceName_};
        if (registeredProducts.end() != registeredProducts.find(entry)) {
          //ignore registration of items if in both begin and end transitions for now
          // This is to work around ExternalLHEProducer
          continue;
        } else {
          registeredProducts.insert(entry);
        }
      }

      TypeWithDict type(p->typeID_.typeInfo());
      BranchDescription pdesc(branchType,
                              iDesc.moduleLabel(),
                              iDesc.processName(),
                              p->typeID_.userClassName(),
                              p->typeID_.friendlyClassName(),
                              p->productInstanceName_,
                              iDesc.moduleName(),
                              iDesc.parameterSetID(),
                              type,
                              true,
                              isEndTransition(p->transition_));
      if (p->aliasType_ == TypeLabelItem::AliasType::kSwitchAlias) {
        if (p->branchAlias_.empty()) {
          throw edm::Exception(edm::errors::LogicError)
              << "Branch alias type has been set to SwitchAlias, but the alias content is empty.\n"
              << "Please report this error to the FWCore developers";
        }
        pdesc.setSwitchAliasModuleLabel(p->branchAlias_);
      }
      if (p->isTransform_) {
        pdesc.setOnDemand(true);
        pdesc.setIsTransform(true);
      }
      setIsMergeable(pdesc);

      if (pdesc.transient()) {
        if (!checkDictionary(missingDictionaries, pdesc.wrappedName(), pdesc.wrappedType())) {
          // It is should be impossible to get here, because the only way to
          // make it transient is in the line that causes the wrapped dictionary
          // to be created. Just to be safe I leave this check here ...
          producedTypes.emplace_back(pdesc.className());
          continue;
        }
      } else {
        // also check constituents of wrapped types if it is not transient
        if (!checkClassDictionaries(missingDictionaries, pdesc.wrappedName(), pdesc.wrappedType())) {
          producedTypes.emplace_back(pdesc.className());
          continue;
        }
      }
      if (!p->branchAlias_.empty())
        pdesc.insertBranchAlias(p->branchAlias_);
      iReg.addProduct(pdesc, iIsListener);
    }

    if (!missingDictionaries.empty()) {
      std::string context("Calling ProductRegistryHelper::addToRegistry, checking dictionaries for produced types");
      throwMissingDictionariesException(missingDictionaries, context, producedTypes);
    }
  }
}  // namespace edm
