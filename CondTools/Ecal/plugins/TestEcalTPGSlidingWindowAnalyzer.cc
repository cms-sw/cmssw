#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGSlidingWindowHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalTPGSlidingWindowHandler> ExTestEcalTPGSlidingWindowAnalyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGSlidingWindowAnalyzer);
