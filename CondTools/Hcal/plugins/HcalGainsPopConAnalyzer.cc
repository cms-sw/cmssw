#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalGainsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<HcalGainsHandler> HcalGainsPopConAnalyzer;


DEFINE_FWK_MODULE(HcalGainsPopConAnalyzer);
