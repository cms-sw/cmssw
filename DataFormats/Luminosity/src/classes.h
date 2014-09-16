#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Luminosity/interface/LumiInfoRunHeader.h"
#include "DataFormats/Luminosity/interface/LumiInfo.h"

namespace DataFormats_Luminosity {
   struct dictionary {
      edm::Wrapper<LumiInfoRunHeader> lumiinforunheaderobj;
      edm::Wrapper<LumiInfo> lumiinfoobj;
   };
}
