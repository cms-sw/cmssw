#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Utilities/interface/typelookup.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "RecoMuon/TrackingTools/plugins/MuonErrorMatrixAdjuster.h"


DEFINE_FWK_MODULE(MuonErrorMatrixAdjuster);
