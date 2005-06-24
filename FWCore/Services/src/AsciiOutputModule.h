/*----------------------------------------------------------------------
$Id: AsciiOutputModule.h,v 1.3 2005/06/09 01:53:38 wmtan Exp $
----------------------------------------------------------------------*/

#include <ostream>

#include "FWCore/CoreFramework/interface/CoreFrameworkfwd.h"
#include "FWCore/CoreFramework/interface/OutputModule.h"

namespace edm {
  class ParameterSet;
  class AsciiOutputModule : public OutputModule {
  public:
    // We do not take ownership of passed stream.
    explicit AsciiOutputModule(ParameterSet const& pset, std::ostream* os = &std::cout);
    virtual ~AsciiOutputModule();
    virtual void write(const EventPrincipal& e);

  private:
    int prescale_;
    int verbosity_;
    int counter_;
    std::ostream* pout_;
  };
}
