#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoJets/JetAnalyzers/interface/JetPlotsExample.h"
#include "RecoJets/JetAnalyzers/interface/JetAnalyzer.h"
#include "RecoJets/JetAnalyzers/interface/JetValidation.h"
#include "RecoJets/JetAnalyzers/interface/CaloTowersExample.h"
#include "RecoJets/JetAnalyzers/interface/JetToDigiDump.h"
#include "RecoJets/JetAnalyzers/interface/DijetMass.h"
#include "RecoJets/JetAnalyzers/interface/SimpleJetDump.h"
#include "RecoJets/JetAnalyzers/interface/CorJetsExample.h"
#include "RecoJets/JetAnalyzers/interface/DijetRatio.h"
 
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( JetPlotsExample );
DEFINE_ANOTHER_FWK_MODULE( JetAnalyzer );
DEFINE_ANOTHER_FWK_MODULE( JetValidation );
DEFINE_ANOTHER_FWK_MODULE( CaloTowersExample );
DEFINE_ANOTHER_FWK_MODULE( JetToDigiDump );
DEFINE_ANOTHER_FWK_MODULE( DijetMass );
DEFINE_ANOTHER_FWK_MODULE( SimpleJetDump );
DEFINE_ANOTHER_FWK_MODULE( CorJetsExample );
DEFINE_ANOTHER_FWK_MODULE( DijetRatio );
