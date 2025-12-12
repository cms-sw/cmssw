#include "CondFormats/HcalObjects/interface/HcalInterpolatedPulseMap.h"
#include "CondFormats/DataRecord/interface/HcalInterpolatedPulseMapRcd.h"

#include "CondTools/Hcal/interface/BoostIODBWriter.h"
#include "CondTools/Hcal/interface/BoostIODBReader.h"

#include "FWCore/Framework/interface/MakerMacros.h"

typedef BoostIODBWriter<HcalInterpolatedPulseMap> HcalInterpolatedPulseMapDBWriter;

typedef BoostIODBReader<HcalInterpolatedPulseMap, HcalInterpolatedPulseMapRcd> HcalInterpolatedPulseMapDBReader;

DEFINE_FWK_MODULE(HcalInterpolatedPulseMapDBWriter);
DEFINE_FWK_MODULE(HcalInterpolatedPulseMapDBReader);
