#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/JetExamples/src/MidpointJetProducer.h"
#include "PhysicsTools/JetExamples/src/JetAnalyzer.h"

typedef cms::MidpointJetProducer MidPointJetProducer;
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( MidPointJetProducer );
DEFINE_ANOTHER_FWK_MODULE( JetAnalyzer );

