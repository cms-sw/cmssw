#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoJets/JetAnalyzers/interface/JetPlotsExample.h"
#include "RecoJets/JetAnalyzers/interface/JetAnalyzer.h"
#include "RecoJets/JetAnalyzers/interface/JetValidation.h"
 
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( JetPlotsExample );
DEFINE_ANOTHER_FWK_MODULE( JetAnalyzer );
DEFINE_ANOTHER_FWK_MODULE( JetValidation );
