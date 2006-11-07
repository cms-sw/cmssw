/*----------------------------------------------------------------------
$Id: AsciiOutputModule.h,v 1.2 2005/10/12 02:34:02 wmtan Exp $
----------------------------------------------------------------------*/

#include <iostream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/OutputModule.h"

namespace edm {
  class ParameterSet;
  class AsciiOutputModule : public OutputModule {
  public:
    // We do not take ownership of passed stream.
    explicit AsciiOutputModule(ParameterSet const& pset, std::ostream* os = &std::cout);
    virtual ~AsciiOutputModule();

  private:
    virtual void write(EventPrincipal const& e);
    virtual void endLuminosityBlock(LuminosityBlockPrincipal const&){}
    virtual void endRun(RunPrincipal const&){}
    int prescale_;
    int verbosity_;
    int counter_;
    std::ostream* pout_;
  };
}
