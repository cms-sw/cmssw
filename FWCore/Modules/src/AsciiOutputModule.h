/*----------------------------------------------------------------------
$Id: AsciiOutputModule.h,v 1.3 2006/11/07 18:06:54 wmtan Exp $
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
    virtual void writeLuminosityBlock(LuminosityBlockPrincipal const&){}
    virtual void writeRun(RunPrincipal const&){}
    int prescale_;
    int verbosity_;
    int counter_;
    std::ostream* pout_;
  };
}
