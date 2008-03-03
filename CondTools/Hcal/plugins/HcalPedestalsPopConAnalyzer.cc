#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalPedestalsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<HcalPedestalsHandler> HcalPedestalsPopConAnalyzer;


DEFINE_FWK_MODULE(HcalPedestalsPopConAnalyzer);
