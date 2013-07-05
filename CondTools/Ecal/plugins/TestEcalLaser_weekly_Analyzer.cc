
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalLaser_weekly_Handler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalLaser_weekly_Handler> ExTestEcalLaser_weekly_Analyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalLaser_weekly_Analyzer);
