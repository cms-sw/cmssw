/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProductRegistryHelper.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Utilities/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

#include <vector>

namespace edm {
  ProductRegistryHelper::~ProductRegistryHelper() { }

  ProductRegistryHelper::TypeLabelList & ProductRegistryHelper::typeLabelList() {
    return typeLabelList_;
  }

  void
  ProductRegistryHelper::addToRegistry(TypeLabelList::const_iterator const& iBegin,
                                       TypeLabelList::const_iterator const& iEnd,
                                       ModuleDescription const& iDesc,
                                       ProductRegistry& iReg,
                                       bool iIsListener) {

    std::vector<std::string> missingDictionaries;

    for(TypeLabelList::const_iterator p = iBegin; p != iEnd; ++p) {

      if (!checkDictionary(missingDictionaries, p->typeID_)) {
        checkDictionaryOfWrappedType(missingDictionaries, p->typeID_);
        continue;
      }

      TypeWithDict type(p->typeID_.typeInfo());
      BranchDescription pdesc(p->branchType_,
                              iDesc.moduleLabel(),
                              iDesc.processName(),
                              p->typeID_.userClassName(),
                              p->typeID_.friendlyClassName(),
                              p->productInstanceName_,
                              iDesc.moduleName(),
                              iDesc.parameterSetID(),
                              type);

      if (!checkClassDictionaries(missingDictionaries, pdesc.wrappedName(), pdesc.wrappedType())) {
        continue;
      }

      if (!p->branchAlias_.empty()) pdesc.insertBranchAlias(p->branchAlias_);
      iReg.addProduct(pdesc, iIsListener);
    }

    if (!missingDictionaries.empty()) {
      std::string context("Calling ProductRegistryHelper::addToRegistry, checking dictionaries for produced types");
      throwMissingDictionariesException(missingDictionaries, context);
    }
  }
}
