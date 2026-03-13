#include "CondFormats/DataRecord/interface/HcalPulseDelaysRcd.h"
#include "CondTools/Hcal/interface/EventSetupPayloadPopConAnalyzer.h"
#include "CondTools/Hcal/interface/HcalPulseDelaysHandler.h"
#include "FWCore/Framework/interface/MakerMacros.h"

typedef EventSetupPayloadPopConAnalyzer<HcalPulseDelaysHandler, HcalPulseDelaysRcd> HcalPulseDelaysPopConAnalyzer;

DEFINE_FWK_MODULE(HcalPulseDelaysPopConAnalyzer);
