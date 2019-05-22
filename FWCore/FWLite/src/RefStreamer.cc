#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "TClass.h"
#include <cassert>
#include <ostream>

class TBuffer;

namespace fwlite {
  edm::EDProductGetter const* setRefStreamer(edm::EDProductGetter const* ep) {
    return edm::EDProductGetter::switchProductGetter(ep);
  }
}  // namespace fwlite
