#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalDAQHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalDAQHandler> ExTestEcalDAQAnalyzer;


//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalDAQAnalyzer);
