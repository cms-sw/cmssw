#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoTBCalo/EcalTBHodoscopeReconstructor/interface/EcalTBHodoscopeRecInfoProducer.h"
#include "RecoTBCalo/EcalTBHodoscopeReconstructor/interface/EcalTBHodoscopeRawInfoDumper.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( EcalTBHodoscopeRecInfoProducer );
DEFINE_ANOTHER_FWK_MODULE( EcalTBHodoscopeRawInfoDumper );

