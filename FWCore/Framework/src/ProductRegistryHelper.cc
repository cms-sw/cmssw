/*----------------------------------------------------------------------

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/ProductRegistryHelper.h"
#include "FWCore/PluginManager/interface/PluginCapabilities.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/Utilities/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "TClass.h"

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
      // This should load the dictionary if not already loaded.
      TClass::GetClass(p->typeID_.typeInfo());
      if(!hasDictionary(p->typeID_.typeInfo())) {
        // a second attempt to load
        edmplugin::PluginCapabilities::get()->tryToLoad(prefix + p->typeID_.userClassName());
      }
      if(!hasDictionary(p->typeID_.typeInfo())) {
        throw Exception(errors::DictionaryNotFound)
           << "No data dictionary found for class:\n\n"
           <<  p->typeID_.className()
           << "\nMost likely the dictionary was never generated,\n"
           << "but it may be that it was generated in the wrong package.\n"
           << "Please add (or move) the specification\n"
           << "<class name=\"whatever\"/>\n"
           << "to the appropriate classes_def.xml file.\n"
           << "If the class is a template instance, you may need\n"
           << "to define a dummy variable of this type in classes.h.\n"
           << "Also, if this class has any transient members,\n"
           << "you need to specify them in classes_def.xml.";
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
      if (!p->branchAlias_.empty()) pdesc.insertBranchAlias(p->branchAlias_);
      iReg.addProduct(pdesc, iIsListener);
    }//for
  }
}
