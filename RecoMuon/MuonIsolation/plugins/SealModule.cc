#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "RecoMuon/MuonIsolation/interface/MuIsoExtractor.h"
#include "RecoMuon/MuonIsolation/interface/MuIsoExtractorFactory.h"
#include "RecoMuon/MuonIsolation/interface/TrackExtractor.h"
#include "RecoMuon/MuonIsolation/interface/CaloExtractor.h"
#include "RecoMuon/MuonIsolation/interface/CaloExtractorByAssociator.h"
#include "RecoMuon/MuonIsolation/interface/ExtractorFromDeposits.h"
DEFINE_EDM_PLUGIN(MuIsoExtractorFactory, muonisolation::TrackExtractor, "TrackExtractor");
DEFINE_EDM_PLUGIN(MuIsoExtractorFactory, muonisolation::CaloExtractor, "CaloExtractor");
DEFINE_EDM_PLUGIN(MuIsoExtractorFactory, muonisolation::CaloExtractorByAssociator, "CaloExtractorByAssociator");
DEFINE_EDM_PLUGIN(MuIsoExtractorFactory, muonisolation::ExtractorFromDeposits, "ExtractorFromDeposits");
