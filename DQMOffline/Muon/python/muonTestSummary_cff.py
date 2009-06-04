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
                             numMatchedExpected = cms.double(4.0),
                             expMolteplicityGlb = cms.double(0.04),
                             expMolteplicityTk = cms.double(0.02),
                             expMolteplicitySta = cms.double(0.8)
                             )
