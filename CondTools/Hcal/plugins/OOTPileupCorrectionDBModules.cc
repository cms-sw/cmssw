#include "CondFormats/HcalObjects/interface/OOTPileupCorrectionColl.h"
#include "CondFormats/DataRecord/interface/HcalOOTPileupCorrectionRcd.h"

#include "CondTools/Hcal/interface/BoostIODBWriter.h"
#include "CondTools/Hcal/interface/BoostIODBReader.h"

#include "FWCore/Framework/interface/MakerMacros.h"

typedef BoostIODBWriter<OOTPileupCorrectionColl> OOTPileupCorrectionDBWriter;
typedef BoostIODBReader<OOTPileupCorrectionColl,HcalOOTPileupCorrectionRcd> OOTPileupCorrectionDBReader;

DEFINE_FWK_MODULE(OOTPileupCorrectionDBWriter);
DEFINE_FWK_MODULE(OOTPileupCorrectionDBReader);
