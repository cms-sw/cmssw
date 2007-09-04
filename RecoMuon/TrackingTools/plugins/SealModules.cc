#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "RecoMuon/TrackingTools/plugins/MuonErrorMatrixAdjuster.h"
#include "RecoMuon/TrackingTools/plugins/MuonErrorMatrixAnalyzer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MuonErrorMatrixAdjuster);
DEFINE_ANOTHER_FWK_MODULE(MuonErrorMatrixAnalyzer);
