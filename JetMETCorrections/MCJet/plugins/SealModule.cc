#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CaloJetResponse.h"
#include "CorJetResponse.h"
#include "SimJetResponseAnalysis.h"

using cms::CaloJetResponse;
using cms::CorJetResponse;
using cms::SimJetResponseAnalysis;

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(CaloJetResponse);
DEFINE_ANOTHER_FWK_MODULE(CorJetResponse);
DEFINE_ANOTHER_FWK_MODULE(SimJetResponseAnalysis);

/*
#include "CondCore/PluginSystem/interface/registration_macros.h"
DEFINE_SEAL_MODULE();

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"

using namespace cms;

#include "CaloJetResponse.h"
DEFINE_ANOTHER_FWK_MODULE(CaloJetResponse);
#include "CorJetResponse.h"
DEFINE_ANOTHER_FWK_MODULE(CorJetResponse);
#include "SimJetResponseAnalysis.h"
DEFINE_ANOTHER_FWK_MODULE(SimJetResponseAnalysis);
*/
