/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProductRegistryHelper.h"
#include "FWCore/PluginManager/interface/PluginCapabilities.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Utilities/interface/DictionaryTools.h"

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
    std::string const& prefix = dictionaryPlugInPrefix();
    for(TypeLabelList::const_iterator p = iBegin; p != iEnd; ++p) {
      if(!p->typeID_.hasDictionary()) {
        //attempt to load
        edmplugin::PluginCapabilities::get()->tryToLoad(prefix + p->typeID_.userClassName());
      }
      BranchDescription pdesc(p->branchType_,
                              iDesc.moduleLabel(),
                              iDesc.processName(),
                              p->typeID_.userClassName(),
                              p->typeID_.friendlyClassName(),
                              p->productInstanceName_,
                              iDesc.moduleName(),
                              iDesc.parameterSetID(),
                              p->typeID_);
      if (!p->branchAlias_.empty()) pdesc.branchAliases().insert(p->branchAlias_);
      iReg.addProduct(pdesc, iIsListener);
    }//for
  }
}
