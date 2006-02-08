#ifndef Framework_GenericInputSource_h
#define Framework_GenericInputSource_h

/*----------------------------------------------------------------------
$Id: GenericInputSource.h,v 1.5 2006/01/18 00:38:44 wmtan Exp $
----------------------------------------------------------------------*/

//#include <memory>

#include "FWCore/Framework/interface/InputSource.h"
#include "DataFormats/Common/interface/ModuleDescription.h"
#include "FWCore/Framework/interface/ProductRegistryHelper.h"

namespace edm {
  class InputSourceDescription;
  class GenericInputSource : public InputSource, public ProductRegistryHelper {
  public:
    explicit GenericInputSource(InputSourceDescription const& desc);
    virtual ~GenericInputSource();

  protected:
    ModuleDescription const& module() const {return module_;}

  private:
    virtual void addToReg(ModuleDescription const& md);
    ModuleDescription module_;
  };
}
#endif
