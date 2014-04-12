#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGWeightGroupHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalTPGWeightGroupHandler> ExTestEcalTPGWeightGroupAnalyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGWeightGroupAnalyzer);
