#include "CondFormats/CastorObjects/interface/CastorElectronicsMap.h"
#include "CondFormats/DataRecord/interface/CastorElectronicsMapRcd.h"
#include "CondTools/Hcal/interface/EventSetupPayloadPopConAnalyzer.h"
#include "CondTools/Hcal/interface/CastorElectronicsMapHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef EventSetupPayloadPopConAnalyzer<CastorElectronicsMapHandler, CastorElectronicsMapRcd>
    CastorElectronicsMapPopConAnalyzer;

DEFINE_FWK_MODULE(CastorElectronicsMapPopConAnalyzer);
