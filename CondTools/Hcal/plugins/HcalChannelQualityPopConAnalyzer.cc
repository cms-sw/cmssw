#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalChannelQualityHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<HcalChannelQualityHandler> HcalChannelQualityPopConAnalyzer;


DEFINE_FWK_MODULE(HcalChannelQualityPopConAnalyzer);
