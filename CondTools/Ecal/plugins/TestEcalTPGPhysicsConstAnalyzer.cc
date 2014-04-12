#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGPhysicsConstHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalTPGPhysicsConstHandler> ExTestEcalTPGPhysicsConstAnalyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGPhysicsConstAnalyzer);
