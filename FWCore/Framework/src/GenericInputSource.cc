/*----------------------------------------------------------------------
$Id: GenericInputSource.cc,v 1.4 2006/01/07 00:38:14 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/GenericInputSource.h"

namespace edm {
  GenericInputSource::GenericInputSource(InputSourceDescription const& desc) :
    InputSource(desc),
    ProductRegistryHelper(),
    module_()
  { }

  GenericInputSource::~GenericInputSource() {
  }

  void
  GenericInputSource::addToReg(ModuleDescription const& md) {
    module_ = md;
    if (!typeLabelList().empty()) {
      ProductRegistryHelper::addToRegistry(typeLabelList().begin(), typeLabelList().end(), md, productRegistry());
    }
  }
}
