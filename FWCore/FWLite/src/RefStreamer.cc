#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "TClass.h"
#include <cassert>
#include <ostream>

class TBuffer;

namespace fwlite {
  edm::EDProductGetter const* setRefStreamer(edm::EDProductGetter const* ep) {
    {
      TClass* cl = TClass::GetClass("edm::RefCore");
      TClassStreamer* st = cl->GetStreamer();
      if (st == 0) {
        cl->AdoptStreamer(new edm::RefCoreStreamer());
      }
    }
    {
      TClass* cl = TClass::GetClass("edm::RefCoreWithIndex");
      TClassStreamer* st = cl->GetStreamer();
      if (st == 0) {
        cl->AdoptStreamer(new edm::RefCoreWithIndexStreamer());
      }
    }
    return edm::EDProductGetter::switchProductGetter(ep);
  }
}
