import FWCore.ParameterSet.Config as cms

muTestSummary = cms.EDFilter("MuonTestSummary",
                             #chose between "ppLike" or "cosmics"
                             dataSample = cms.untracked.string('cosmics'),
                             # tests parameters
                             etaSpread = cms.double(0.4), 
                             phiSpread = cms.double(0.4),
                             chi2Fraction = cms.double(0.3),
                             chi2Spread = cms.double(0.1)
                             )
