#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoTBCalo/EcalTBTDCReconstructor/interface/EcalTBTDCRecInfoProducer.h"
#include "RecoTBCalo/EcalTBTDCReconstructor/interface/EcalTBH2TDCRecInfoProducer.h"
#include "RecoTBCalo/EcalTBTDCReconstructor/interface/EcalTBTDCRawInfoDumper.h"


DEFINE_FWK_MODULE( EcalTBTDCRecInfoProducer );
DEFINE_FWK_MODULE( EcalTBH2TDCRecInfoProducer );
DEFINE_FWK_MODULE( EcalTBTDCRawInfoDumper );

