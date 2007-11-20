#include "IOPool/Common/interface/ClassFiller.h"
#include "Cintex/Cintex.h"
//#include "TH1.h"

namespace edm {
  // ---------------------
  void ClassFiller() {
    // We may want to do this in the future but for now we have to give people
    // time to adjust their code...
    //    TH1::AddDirectory(kFALSE);
    ROOT::Cintex::Cintex::Enable();
  }
}
