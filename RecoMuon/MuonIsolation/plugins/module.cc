#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractor.h"
#include "PhysicsTools/IsolationAlgos/interface/IsoDepositExtractorFactory.h"
#include "TrackExtractor.h"
#include "PixelTrackExtractor.h"
#include "CaloExtractor.h"
#include "CaloExtractorByAssociator.h"
#include "JetExtractor.h"
#include "ExtractorFromDeposits.h"
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactoryFromHelper, muonisolation::TrackExtractor, "TrackExtractor");
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactoryFromHelper, muonisolation::PixelTrackExtractor, "PixelTrackExtractor");
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactoryFromHelper, muonisolation::CaloExtractor, "CaloExtractor");
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactoryFromHelper, muonisolation::CaloExtractorByAssociator, "CaloExtractorByAssociator");
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactoryFromHelper, muonisolation::JetExtractor, "JetExtractor");
DEFINE_EDM_PLUGIN(IsoDepositExtractorFactoryFromHelper, muonisolation::ExtractorFromDeposits, "ExtractorFromDeposits");

#include "RecoMuon/MuonIsolation/interface/MuonIsolatorFactory.h"
#include "RecoMuon/MuonIsolation/interface/SimpleCutsIsolator.h"
#include "CutsIsolatorWithCorrection.h"

DEFINE_EDM_PLUGIN(MuonIsolatorFactory, SimpleCutsIsolator, "SimpleCutsIsolator");
DEFINE_EDM_PLUGIN(MuonIsolatorFactory, CutsIsolatorWithCorrection, "CutsIsolatorWithCorrection");
