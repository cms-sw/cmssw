#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGLinConstHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalTPGLinConstHandler> ExTestEcalTPGLinConstAnalyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGLinConstAnalyzer);
