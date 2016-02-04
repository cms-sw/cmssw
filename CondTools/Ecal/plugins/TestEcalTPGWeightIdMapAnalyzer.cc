#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGWeightIdMapHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalTPGWeightIdMapHandler> ExTestEcalTPGWeightIdMapAnalyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGWeightIdMapAnalyzer);
