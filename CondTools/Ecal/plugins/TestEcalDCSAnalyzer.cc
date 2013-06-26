#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalDCSHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalDCSHandler> ExTestEcalDCSAnalyzer;


//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalDCSAnalyzer);
