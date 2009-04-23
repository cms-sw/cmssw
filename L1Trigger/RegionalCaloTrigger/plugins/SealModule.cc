#include "FWCore/Framework/interface/MakerMacros.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTProducer.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTInputProducer.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTTestAnalyzer.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTSaveInput.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTRelValAnalyzer.h"
#include "L1Trigger/RegionalCaloTrigger/interface/L1RCTLutWriter.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(L1RCTProducer);
DEFINE_ANOTHER_FWK_MODULE(L1RCTInputProducer);
DEFINE_ANOTHER_FWK_MODULE(L1RCTTestAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(L1RCTSaveInput);
DEFINE_ANOTHER_FWK_MODULE(L1RCTRelValAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(L1RCTLutWriter);
