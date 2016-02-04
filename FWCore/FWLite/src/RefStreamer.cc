#include "DataFormats/Common/interface/RefCore.h"
#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "TROOT.h"
#include <assert.h>
#include <ostream>

class TBuffer;

namespace fwlite {
  edm::EDProductGetter const* setRefStreamer(edm::EDProductGetter const* ep) {
    {
      TClass* cl = gROOT->GetClass("edm::RefCore::CheckTransientOnWrite");
      TClassStreamer* st = cl->GetStreamer();
      if (st == 0) {
        cl->AdoptStreamer(new edm::RefCoreCheckTransientOnWriteStreamer());
      }
    }
    return edm::EDProductGetter::switchProductGetter(ep);
  }
}
