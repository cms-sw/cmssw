#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalPulseSymmCovariancesHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::EcalPulseSymmCovariancesHandler> ExTestEcalPulseSymmCovariancesAnalyzer;

//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalPulseSymmCovariancesAnalyzer);
