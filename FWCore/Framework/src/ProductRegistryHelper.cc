/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProductRegistryHelper.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Utilities/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

#include <vector>
#include <typeindex>

namespace edm {
  ProductRegistryHelper::~ProductRegistryHelper() { }

  ProductRegistryHelper::TypeLabelList const& ProductRegistryHelper::typeLabelList() const {
    return typeLabelList_;
  }

  void
  ProductRegistryHelper::addToRegistry(TypeLabelList::const_iterator const& iBegin,
                                       TypeLabelList::const_iterator const& iEnd,
                                       ModuleDescription const& iDesc,
                                       ProductRegistry& iReg,
                                       bool iIsListener) {

    std::vector<std::string> missingDictionaries;
    std::vector<std::string> producedTypes;
    std::set<std::tuple<BranchType,std::type_index,std::string>> registeredProducts;

    for(TypeLabelList::const_iterator p = iBegin; p != iEnd; ++p) {

      if (!checkDictionary(missingDictionaries, p->typeID_)) {
        checkDictionaryOfWrappedType(missingDictionaries, p->typeID_);
        producedTypes.emplace_back(p->typeID_.className());
        continue;
      }
      auto branchType = convertToBranchType(p->transition_);
      if(branchType != InEvent) {
        std::tuple<BranchType, std::type_index, std::string> entry{ branchType,p->typeID_.typeInfo(),p->productInstanceName_};
        if(registeredProducts.end() != registeredProducts.find(entry) ) {
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
      if (!p->branchAlias_.empty()) pdesc.insertBranchAlias(p->branchAlias_);
      iReg.addProduct(pdesc, iIsListener);
    }

    if (!missingDictionaries.empty()) {
      std::string context("Calling ProductRegistryHelper::addToRegistry, checking dictionaries for produced types");
      throwMissingDictionariesException(missingDictionaries, context, producedTypes);
    }
  }
}
