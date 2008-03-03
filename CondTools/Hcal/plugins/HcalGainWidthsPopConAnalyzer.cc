#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalGainWidthsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<HcalGainWidthsHandler> HcalGainWidthsPopConAnalyzer;


DEFINE_FWK_MODULE(HcalGainWidthsPopConAnalyzer);
