#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Luminosity/interface/LumiSummaryRunHeader.h"
#include "DataFormats/Luminosity/interface/LumiSummary.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"
#include "DataFormats/Luminosity/interface/LumiInfoRunHeader.h"
#include "DataFormats/Luminosity/interface/LumiInfo.h"
#include "DataFormats/Luminosity/interface/BeamCurrentInfo.h"
#include "DataFormats/Luminosity/interface/PCC.h"

namespace DataFormats_Luminosity {
   struct dictionary {
      edm::Wrapper<LumiSummaryRunHeader> lumisummaryrunheaderobj;
      edm::Wrapper<LumiSummary> lumisummaryobj;
      edm::Wrapper<LumiDetails> lumidetailsobj;
      edm::Wrapper<LumiInfoRunHeader> lumiinforunheaderobj;
      edm::Wrapper<LumiInfo> lumiinfoobj;
      edm::Wrapper<BeamCurrentInfo> beamcurrentinfoobj;
      reco::PCC b;
      edm::Wrapper<reco::PCC> b_w;
   };
}
