/*----------------------------------------------------------------------
$Id: AsciiOutputModule.h,v 1.7 2005/07/28 19:35:57 wmtan Exp $
----------------------------------------------------------------------*/

#include <iostream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/OutputModule.h"

namespace edm {
  class ParameterSet;
  class AsciiOutputModule : public OutputModule {
  public:
    // We do not take ownership of passed stream.
    explicit AsciiOutputModule(ParameterSet const& pset, ProductRegistry const&, std::ostream* os = &std::cout);
    virtual ~AsciiOutputModule();
    virtual void write(const EventPrincipal& e);

  private:
    int prescale_;
    int verbosity_;
    int counter_;
    std::ostream* pout_;
  };
}
