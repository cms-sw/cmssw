import FWCore.ParameterSet.Config as cms

import DQMOffline.Muon.muonTestSummary_cfi
muonCosmicTestSummary = DQMOffline.Muon.muonTestSummary_cfi.muonTestSummary.clone()
muonCosmicTestSummary.etaExpected = cms.double(0.5)
muonCosmicTestSummary.phiExpected = cms.double(0.01)
muonCosmicTestSummary.expMultiplicityGlb_min = cms.double(0.)
muonCosmicTestSummary.expMultiplicityGlb_max = cms.double(0.1)
muonCosmicTestSummary.expMultiplicityTk_min = cms.double(0.)
muonCosmicTestSummary.expMultiplicityTk_max = cms.double(0.045)
muonCosmicTestSummary.expMultiplicitySta_min = cms.double(0.75)
muonCosmicTestSummary.expMultiplicitySta_max = cms.double(0.95)

