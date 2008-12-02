import FWCore.ParameterSet.Config as cms

muonTestSummary = cms.EDFilter("MuonTestSummary",
                             # tests parameters
                             etaExpected = cms.double(0.6),
                             phiExpected = cms.double(0.01),
                             etaSpread = cms.double(0.3), 
                             phiSpread = cms.double(0.01),
                             chi2Fraction = cms.double(0.4),
                             chi2Spread = cms.double(0.2),
                             resEtaSpread_tkGlb = cms.double(0.001),
                             resEtaSpread_glbSta = cms.double(0.05),
                             resPhiSpread_tkGlb = cms.double(0.001),
                             resPhiSpread_glbSta = cms.double(0.05),
                             numMatchedExpected = cms.double(4.0),
                             sigmaResSegmTrackExp = cms.double(1.0)
                             )
