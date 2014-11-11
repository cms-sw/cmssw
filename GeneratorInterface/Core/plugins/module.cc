#include "FWCore/Framework/interface/MakerMacros.h"

#include "GeneratorInterface/Core/interface/GenFilterEfficiencyProducer.h"
DEFINE_FWK_MODULE(GenFilterEfficiencyProducer);

#include "GeneratorInterface/Core/interface/GenFilterEfficiencyAnalyzer.h"
DEFINE_FWK_MODULE(GenFilterEfficiencyAnalyzer);


#include "GeneratorInterface/Core/interface/GenXSecAnalyzer.h"
DEFINE_FWK_MODULE(GenXSecAnalyzer);
