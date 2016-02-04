#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGBadStripHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalTPGBadStripHandler> ExTestEcalTPGBadStripAnalyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGBadStripAnalyzer);
