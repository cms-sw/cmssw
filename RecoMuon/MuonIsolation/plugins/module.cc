#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractorFactory.h"
#include "TrackExtractor.h"
#include "PixelTrackExtractor.h"
#include "CaloExtractor.h"
#include "CaloExtractorByAssociator.h"
#include "JetExtractor.h"
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactory, muonisolation::TrackExtractor, "TrackExtractor");
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactory, muonisolation::PixelTrackExtractor, "PixelTrackExtractor");
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactory, muonisolation::CaloExtractor, "CaloExtractor");
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactory, muonisolation::CaloExtractorByAssociator, "CaloExtractorByAssociator");
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactory, muonisolation::JetExtractor, "JetExtractor");

#include "RecoMuon/MuonIsolation/interface/MuonIsolatorFactory.h"
#include "RecoMuon/MuonIsolation/interface/SimpleCutsIsolator.h"
#include "CutsIsolatorWithCorrection.h"

DEFINE_EDM_PLUGIN(MuonIsolatorFactory, SimpleCutsIsolator, "SimpleCutsIsolator");
DEFINE_EDM_PLUGIN(MuonIsolatorFactory, CutsIsolatorWithCorrection, "CutsIsolatorWithCorrection");
