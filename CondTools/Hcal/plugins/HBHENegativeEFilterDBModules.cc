#include "CondFormats/HcalObjects/interface/HBHENegativeEFilter.h"
#include "CondFormats/DataRecord/interface/HBHENegativeEFilterRcd.h"

#include "CondTools/Hcal/interface/BoostIODBWriter.h"
#include "CondTools/Hcal/interface/BoostIODBReader.h"

#include "FWCore/Framework/interface/MakerMacros.h"

typedef BoostIODBWriter<HBHENegativeEFilter> HBHENegativeEFilterDBWriter;
typedef BoostIODBReader<HBHENegativeEFilter,HBHENegativeEFilterRcd> HBHENegativeEFilterDBReader;

DEFINE_FWK_MODULE(HBHENegativeEFilterDBWriter);
DEFINE_FWK_MODULE(HBHENegativeEFilterDBReader);
