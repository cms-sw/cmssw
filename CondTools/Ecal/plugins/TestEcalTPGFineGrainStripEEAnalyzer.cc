#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGFineGrainStripEEHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalTPGFineGrainStripEEHandler> ExTestEcalTPGFineGrainStripEEAnalyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGFineGrainStripEEAnalyzer);
