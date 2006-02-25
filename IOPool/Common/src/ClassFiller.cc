#include "IOPool/Common/interface/ClassFiller.h"
#include "Cintex/Cintex.h"

namespace edm {
  // ---------------------
  void ClassFiller() {
    ROOT::Cintex::Cintex::Enable();
  }
}
