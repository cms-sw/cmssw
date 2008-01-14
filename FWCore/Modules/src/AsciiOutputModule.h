/*----------------------------------------------------------------------
$Id: AsciiOutputModule.h,v 1.4 2008/01/05 05:28:53 wmtan Exp $
----------------------------------------------------------------------*/

#include <iostream>
#include <string>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/OutputModule.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
//#include "FWCore/MessageLogger/src/MessageLogger.cc"


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
