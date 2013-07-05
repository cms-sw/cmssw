
#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalLaser_weekly_Linearization_Check.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalLaser_weekly_Linearization_Check> ExTestEcalLaser_weekly_Linearization_Check;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalLaser_weekly_Linearization_Check);
