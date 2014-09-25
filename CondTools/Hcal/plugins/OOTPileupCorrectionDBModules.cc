#include "CondFormats/HcalObjects/interface/OOTPileupCorrectionColl.h"
#include "CondFormats/HcalObjects/interface/OOTPileupCorrectionMapColl.h"
#include "CondFormats/DataRecord/interface/HcalOOTPileupCorrectionRcd.h"
#include "CondFormats/DataRecord/interface/HcalOOTPileupCompatibilityRcd.h"
#include "CondFormats/DataRecord/interface/HcalOOTPileupCorrectionMapCollRcd.h"

#include "CondTools/Hcal/interface/BoostIODBWriter.h"
#include "CondTools/Hcal/interface/BoostIODBReader.h"

#include "FWCore/Framework/interface/MakerMacros.h"

typedef BoostIODBWriter<OOTPileupCorrectionColl> OOTPileupCorrectionDBWriter;
typedef BoostIODBWriter<OOTPileupCorrectionMapColl> OOTPileupCorrectionDBV1Writer;

typedef BoostIODBReader<OOTPileupCorrectionColl,HcalOOTPileupCorrectionRcd> OOTPileupCorrectionDBReader;
typedef BoostIODBReader<OOTPileupCorrectionColl,HcalOOTPileupCompatibilityRcd> OOTPileupCompatibilityDBReader;
typedef BoostIODBReader<OOTPileupCorrectionMapColl,HcalOOTPileupCorrectionMapCollRcd> OOTPileupCorrectionDBV1Reader;

DEFINE_FWK_MODULE(OOTPileupCorrectionDBWriter);
DEFINE_FWK_MODULE(OOTPileupCorrectionDBV1Writer);
DEFINE_FWK_MODULE(OOTPileupCorrectionDBReader);
DEFINE_FWK_MODULE(OOTPileupCompatibilityDBReader);
DEFINE_FWK_MODULE(OOTPileupCorrectionDBV1Reader);
