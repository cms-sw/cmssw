
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalLaserHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalLaserHandler> ExTestEcalLaserAnalyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalLaserAnalyzer);
