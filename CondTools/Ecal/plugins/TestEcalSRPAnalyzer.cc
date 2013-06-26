#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalSRPHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalSRPHandler> ExTestEcalSRPAnalyzer;


//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalSRPAnalyzer);
