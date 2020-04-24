#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGLinPed.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::EcalTPGLinPed> ExTestEcalTPGLinPed_Analyzer;

//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGLinPed_Analyzer);
