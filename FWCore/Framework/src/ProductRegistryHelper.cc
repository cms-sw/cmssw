/*----------------------------------------------------------------------
  
$Id: ProductRegistryHelper.cc,v 1.13 2006/11/04 00:35:43 wmtan Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/ModuleDescriptionRegistry.h"
#include "FWCore/Framework/interface/ProductRegistryHelper.h"
#include "DataFormats/Common/interface/ProductRegistry.h"
#include "DataFormats/Common/interface/BranchDescription.h"
#include "DataFormats/Common/interface/ModuleDescription.h"

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
    for (TypeLabelList::const_iterator p = iBegin; p != iEnd; ++p) {
      BranchDescription pdesc(p->branchType_,
                              iDesc.moduleLabel(),
                              iDesc.processName(),
                              p->typeID_.userClassName(),
                              p->typeID_.friendlyClassName(), 
                              p->productInstanceName_,
			      iDesc);
      if (!p->branchAlias_.empty()) pdesc.branchAliases_.insert(p->branchAlias_);
      iReg.addProduct(pdesc, iIsListener);
      ModuleDescriptionRegistry::instance()->insertMapped(iDesc);
    }//for
  }
}
