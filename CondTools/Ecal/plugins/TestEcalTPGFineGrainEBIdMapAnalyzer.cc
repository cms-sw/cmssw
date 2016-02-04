#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGFineGrainEBIdMapHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalTPGFineGrainEBIdMapHandler> ExTestEcalTPGFineGrainEBIdMapAnalyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGFineGrainEBIdMapAnalyzer);
