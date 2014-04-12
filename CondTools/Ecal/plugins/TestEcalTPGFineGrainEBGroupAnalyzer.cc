#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGFineGrainEBGroupHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalTPGFineGrainEBGroupHandler> ExTestEcalTPGFineGrainEBGroupAnalyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGFineGrainEBGroupAnalyzer);
