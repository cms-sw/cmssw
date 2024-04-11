#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTimeCalibHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::EcalTimeCalibHandler> ExTestEcalTimeCalibAnalyzer;

//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTimeCalibAnalyzer);
