#include "CondFormats/HcalObjects/interface/HcalInterpolatedPulseColl.h"
#include "CondFormats/DataRecord/interface/HcalInterpolatedPulseCollRcd.h"

#include "CondTools/Hcal/interface/BoostIODBWriter.h"
#include "CondTools/Hcal/interface/BoostIODBReader.h"

#include "FWCore/Framework/interface/MakerMacros.h"

typedef BoostIODBWriter<HcalInterpolatedPulseColl> HcalInterpolatedPulseDBWriter;

typedef BoostIODBReader<HcalInterpolatedPulseColl,HcalInterpolatedPulseCollRcd> HcalInterpolatedPulseDBReader;

DEFINE_FWK_MODULE(HcalInterpolatedPulseDBWriter);
DEFINE_FWK_MODULE(HcalInterpolatedPulseDBReader);
