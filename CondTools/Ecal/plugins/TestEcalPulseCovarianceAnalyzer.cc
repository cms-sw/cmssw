#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalPulseCovariancesHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::EcalPulseCovariancesHandler> ExTestEcalPulseCovariancesAnalyzer;

//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalPulseCovariancesAnalyzer);
