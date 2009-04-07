import FWCore.ParameterSet.Config as cms

muonTestSummary = cms.EDFilter("MuonTestSummary",
                             # tests parameters
                             etaExpected = cms.double(0.5),
                             phiExpected = cms.double(0.01),
                             chi2Fraction = cms.double(0.4),
                             chi2Spread = cms.double(0.2),
                             resEtaSpread_tkGlb = cms.double(0.001),
                             resEtaSpread_glbSta = cms.double(0.05),
                             resPhiSpread_tkGlb = cms.double(0.001),
                             resPhiSpread_glbSta = cms.double(0.05),
                             resOneOvPSpread_tkGlb = cms.double(0.001),
                             resOneOvPSpread_glbSta = cms.double(0.05),
                             resChargeLimit_tkGlb = cms.double(0.10),
                             resChargeLimit_glbSta = cms.double(0.14),
                             resChargeLimit_tkSta = cms.double(0.18),
                             numMatchedExpected_min = cms.double(3.0),
                             numMatchedExpected_max = cms.double(5.0),
                             matchesFractionDt_min = cms.double(0.05),
                             matchesFractionDt_max = cms.double(0.25),
                             matchesFractionCsc_min = cms.double(0.05),
                             matchesFractionCsc_max = cms.double(0.25),
                             resSegmTrack_min = cms.double(0.9),
                             resSegmTrack_max = cms.double(1.1),
                             expMolteplicityGlb = cms.double(0.04),
                             expMolteplicityTk = cms.double(0.02),
                             expMolteplicitySta = cms.double(0.8)
                             )
