#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalPedestalsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalPedestalsHandler> ExTestEcalPedestalsAnalyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalPedestalsAnalyzer);
