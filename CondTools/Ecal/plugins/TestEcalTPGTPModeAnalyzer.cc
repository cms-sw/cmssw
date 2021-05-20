#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGTPModeHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::EcalTPGTPModeHandler> ExTestEcalTPGTPModeAnalyzer;

//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGTPModeAnalyzer);
