#include "CondCore/PopCon/interface/PopConAnalyzer.h"
#include "CondTools/Ecal/interface/EcalTPGSpikeThresholdHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"



typedef popcon::PopConAnalyzer<popcon::EcalTPGSpikeThresholdHandler> ExTestEcalTPGSpikeAnalyzer;




//define this as a plug-in
DEFINE_FWK_MODULE(ExTestEcalTPGSpikeAnalyzer);
