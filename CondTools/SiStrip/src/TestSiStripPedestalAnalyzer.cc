#include "CondCore/PopCon/interface/PopConAnalyzer.h"
//#include "CondTools/Ecal/interface/EcalPedestalsHandler.h"
#include "CondTools/SiStrip/interface/SiStripConditionHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::SistripConditionsHandler> ExTestEcalPedestalsAnalyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalPedestalsAnalyzer);
