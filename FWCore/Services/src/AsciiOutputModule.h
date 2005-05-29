/*----------------------------------------------------------------------
$Id: AsciiOutputModule.h,v 1.1 2005/05/28 05:10:04 wmtan Exp $
----------------------------------------------------------------------*/

#include <ostream>

#include "FWCore/CoreFramework/interface/OutputModule.h"

namespace edm {
  class EventPrincipal;
  class ParameterSet;
  class AsciiOutputModule : public OutputModule {
  public:
    // We do not take ownership of passed stream.
    explicit AsciiOutputModule(ParameterSet const&, std::ostream* os = &std::cout);
    virtual ~AsciiOutputModule();
    virtual void write(const EventPrincipal& e);

  private:
    std::ostream* pout_;
  };
}
