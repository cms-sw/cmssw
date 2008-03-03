#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalQIEDataHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<HcalQIEDataHandler> HcalQIEDataPopConAnalyzer;


DEFINE_FWK_MODULE(HcalQIEDataPopConAnalyzer);
