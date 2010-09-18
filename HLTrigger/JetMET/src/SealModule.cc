#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/JetMET/interface/HLT2jetGapFilter.h"
#include "HLTrigger/JetMET/interface/HLTAcoFilter.h"
#include "HLTrigger/JetMET/interface/HLTDiJetAveFilter.h"
#include "HLTrigger/JetMET/interface/HLTExclDiJetFilter.h"
#include "HLTrigger/JetMET/interface/HLTForwardBackwardJetsFilter.h"
#include "HLTrigger/JetMET/interface/HLTHcalMETNoiseFilter.h"
#include "HLTrigger/JetMET/interface/HLTHPDFilter.h"
#include "HLTrigger/JetMET/interface/HLTJetVBFFilter.h"
#include "HLTrigger/JetMET/interface/HLTMhtHtFilter.h"
#include "HLTrigger/JetMET/interface/HLTNVFilter.h"
#include "HLTrigger/JetMET/interface/HLTPhi2METFilter.h"
#include "HLTrigger/JetMET/interface/HLTRapGapFilter.h"

DEFINE_FWK_MODULE(HLT2jetGapFilter);
DEFINE_FWK_MODULE(HLTAcoFilter);
DEFINE_FWK_MODULE(HLTDiJetAveFilter);
DEFINE_FWK_MODULE(HLTExclDiJetFilter);
DEFINE_FWK_MODULE(HLTForwardBackwardJetsFilter);
DEFINE_FWK_MODULE(HLTHcalMETNoiseFilter);
DEFINE_FWK_MODULE(HLTHPDFilter);
DEFINE_FWK_MODULE(HLTJetVBFFilter);
DEFINE_FWK_MODULE(HLTMhtHtFilter);
DEFINE_FWK_MODULE(HLTNVFilter);
DEFINE_FWK_MODULE(HLTPhi2METFilter);
DEFINE_FWK_MODULE(HLTRapGapFilter);
