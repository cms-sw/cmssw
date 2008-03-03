#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalRespCorrsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<HcalRespCorrsHandler> HcalRespCorrsPopConAnalyzer;


DEFINE_FWK_MODULE(HcalRespCorrsPopConAnalyzer);
