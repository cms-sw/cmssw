#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalZSThresholdsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<HcalZSThresholdsHandler> HcalZSThresholdsPopConAnalyzer;


DEFINE_FWK_MODULE(HcalZSThresholdsPopConAnalyzer);
