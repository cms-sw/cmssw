
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Luminosity/interface/LumiSummary.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"

namespace {
   struct dictionary {
      edm::Wrapper<LumiSummary> lumisummaryobj;
      edm::Wrapper<LumiDetails> lumidetailsobj;
   };
}
