#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalADCToGeVHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalADCToGeVHandler> ExTestEcalADCToGeVAnalyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalADCToGeVAnalyzer);
