#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalPedestalWidthsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<HcalPedestalWidthsHandler> HcalPedestalWidthsPopConAnalyzer;


DEFINE_FWK_MODULE(HcalPedestalWidthsPopConAnalyzer);
