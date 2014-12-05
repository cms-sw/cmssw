#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalPulseShapesHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalPulseShapesHandler> ExTestEcalPulseShapesAnalyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalPulseShapesAnalyzer);
