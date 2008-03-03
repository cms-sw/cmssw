#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalElectronicsMapHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<HcalElectronicsMapHandler> HcalElectronicsMapPopConAnalyzer;


DEFINE_FWK_MODULE(HcalElectronicsMapPopConAnalyzer);
