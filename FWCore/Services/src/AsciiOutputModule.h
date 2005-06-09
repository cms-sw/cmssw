/*----------------------------------------------------------------------
$Id: AsciiOutputModule.h,v 1.2 2005/06/07 23:47:36 wmtan Exp $
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
    std::ostream* pout_;
  };
}
