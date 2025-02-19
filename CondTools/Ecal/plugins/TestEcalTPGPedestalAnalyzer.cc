#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGPedestalsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalTPGPedestalsHandler> ExTestEcalTPGPedestalsAnalyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGPedestalsAnalyzer);
