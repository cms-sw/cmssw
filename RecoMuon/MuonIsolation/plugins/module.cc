#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "RecoMuon/MuonIsolation/interface/MuIsoExtractor.h"
#include "RecoMuon/MuonIsolation/interface/MuIsoExtractorFactory.h"
#include "TrackExtractor.h"
#include "CaloExtractor.h"
#include "CaloExtractorByAssociator.h"
#include "JetExtractor.h"
#include "ExtractorFromDeposits.h"
DEFINE_EDM_PLUGIN(MuIsoExtractorFactory, muonisolation::TrackExtractor, "TrackExtractor");
DEFINE_EDM_PLUGIN(MuIsoExtractorFactory, muonisolation::CaloExtractor, "CaloExtractor");
DEFINE_EDM_PLUGIN(MuIsoExtractorFactory, muonisolation::CaloExtractorByAssociator, "CaloExtractorByAssociator");
DEFINE_EDM_PLUGIN(MuIsoExtractorFactory, muonisolation::JetExtractor, "JetExtractor");
DEFINE_EDM_PLUGIN(MuIsoExtractorFactory, muonisolation::ExtractorFromDeposits, "ExtractorFromDeposits");
