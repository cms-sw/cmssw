#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalIntercalibHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalIntercalibHandler> ExTestEcalIntercalibAnalyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalIntercalibAnalyzer);
