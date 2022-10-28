import FWCore.ParameterSet.Config as cms

import DQMOffline.Muon.muonTestSummary_cfi
muonCosmicTestSummary = DQMOffline.Muon.muonTestSummary_cfi.muonTestSummary.clone(
    etaExpected = 0.5,
    phiExpected = 0.01,
    expMultiplicityGlb_min = 0.,
    expMultiplicityGlb_max = 0.1,
    expMultiplicityTk_min = 0.,
    expMultiplicityTk_max = 0.045,
    expMultiplicitySta_min = 0.75,
    expMultiplicitySta_max = 0.95
)
