/*----------------------------------------------------------------------
$Id: AsciiOutputModule.h,v 1.5 2008/01/14 16:49:23 chrjones Exp $
----------------------------------------------------------------------*/

#include <iostream>
#include <string>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/OutputModule.h"


namespace edm {
  class ParameterSet;
  class AsciiOutputModule : public OutputModule {
  public:
    // We do not take ownership of passed stream.
    explicit AsciiOutputModule(ParameterSet const& pset);
    virtual ~AsciiOutputModule();

  private:
    virtual void write(EventPrincipal const& e);
    virtual void writeLuminosityBlock(LuminosityBlockPrincipal const&){}
    virtual void writeRun(RunPrincipal const&){}
    int prescale_;
    int verbosity_;
    int counter_;
  };
}
