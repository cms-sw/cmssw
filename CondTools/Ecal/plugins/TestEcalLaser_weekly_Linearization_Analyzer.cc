
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalLaser_weekly_Linearization.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalLaser_weekly_Linearization> ExTestEcalLaser_weekly_Linearization_Analyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalLaser_weekly_Linearization_Analyzer);
