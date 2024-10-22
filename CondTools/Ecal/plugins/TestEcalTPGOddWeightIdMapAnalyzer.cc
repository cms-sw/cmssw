#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGOddWeightIdMapHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::EcalTPGOddWeightIdMapHandler> ExTestEcalTPGOddWeightIdMapAnalyzer;

//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGOddWeightIdMapAnalyzer);
