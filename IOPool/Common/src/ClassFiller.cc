#include "IOPool/Common/interface/ClassFiller.h"
#include "Cintex/Cintex.h"
#include "TH1.h"
#include "G__ci.h"

namespace edm {
  // ---------------------
  void ClassFiller() {
    TH1::AddDirectory(kFALSE);
    ROOT::Cintex::Cintex::Enable();
  }
}
