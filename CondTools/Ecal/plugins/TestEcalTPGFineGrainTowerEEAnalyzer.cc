#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGFineGrainTowerEEHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalTPGFineGrainTowerEEHandler> ExTestEcalTPGFineGrainTowerEEAnalyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGFineGrainTowerEEAnalyzer);
