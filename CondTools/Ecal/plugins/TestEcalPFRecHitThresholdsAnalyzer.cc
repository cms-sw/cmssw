#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalPFRecHitThresholdsHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef popcon::PopConAnalyzer<popcon::EcalPFRecHitThresholdsHandler> ExTestEcalPFRecHitThresholdsAnalyzer;

//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalPFRecHitThresholdsAnalyzer);
