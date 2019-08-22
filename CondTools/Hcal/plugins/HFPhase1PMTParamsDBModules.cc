#include "CondFormats/HcalObjects/interface/HFPhase1PMTParams.h"
#include "CondFormats/DataRecord/interface/HFPhase1PMTParamsRcd.h"

#include "CondTools/Hcal/interface/BoostIODBWriter.h"
#include "CondTools/Hcal/interface/BoostIODBReader.h"

#include "FWCore/Framework/interface/MakerMacros.h"

typedef BoostIODBWriter<HFPhase1PMTParams> HFPhase1PMTParamsDBWriter;
typedef BoostIODBReader<HFPhase1PMTParams, HFPhase1PMTParamsRcd> HFPhase1PMTParamsDBReader;

DEFINE_FWK_MODULE(HFPhase1PMTParamsDBWriter);
DEFINE_FWK_MODULE(HFPhase1PMTParamsDBReader);
