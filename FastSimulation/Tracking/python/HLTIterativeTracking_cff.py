import FWCore.ParameterSet.Config as cms

# watch out here, we assume that
#   - loading FastSimulation.Configuration.Digi_cff.py causes generalTracksAliasInfo.key.value() to return "mix"
#   - loading FastSimulation.Configuration.DataMixerPreMix_cff.py causes generalTracksAliasInfo.key.value() to return "dataMix"
# such that the HLT reads the tracks from the right MixingModule

from FastSimulation.Tracking.GeneralTracksAlias_cfi import generalTracksAliasInfo
_aliasparameters = {generalTracksAliasInfo.key.value():generalTracksAliasInfo.value}

hltIter4Merged = cms.EDAlias(**_aliasparameters)
hltIter2Merged = cms.EDAlias(**_aliasparameters)
hltIter4HighPtMerged = cms.EDAlias(**_aliasparameters)
hltIter2HighPtMerged = cms.EDAlias(**_aliasparameters)
hltIter4Tau3MuMerged = cms.EDAlias(**_aliasparameters)
hltIter4MergedReg = cms.EDAlias(**_aliasparameters)
hltIter2MergedForElectrons = cms.EDAlias(**_aliasparameters)
hltIter2MergedForPhotons = cms.EDAlias(**_aliasparameters)
hltIter2L3MuonMerged = cms.EDAlias(**_aliasparameters)
hltIter2L3MuonMergedReg = cms.EDAlias(**_aliasparameters)
hltIter2MergedForBTag = cms.EDAlias(**_aliasparameters)
hltIter2MergedForTau = cms.EDAlias(**_aliasparameters)
hltIter4MergedForTau = cms.EDAlias(**_aliasparameters)
hltIter2GlbTrkMuonMerged = cms.EDAlias(**_aliasparameters)
hltIter2HighPtTkMuMerged  = cms.EDAlias(**_aliasparameters)
hltIter2HighPtTkMuIsoMerged  = cms.EDAlias(**_aliasparameters)
hltIter2DisplacedJpsiMerged     = cms.EDAlias(**_aliasparameters)
hltIter2DisplacedPsiPrimeMerged = cms.EDAlias(**_aliasparameters)
hltIter2DisplacedNRMuMuMerged   = cms.EDAlias(**_aliasparameters)
hltIter0PFlowTrackSelectionHighPurityForBTag = cms.EDAlias(**_aliasparameters)
hltIter4MergedWithIter012DisplacedJets = cms.EDAlias(**_aliasparameters)

HLTIterativeTrackingIter04 = cms.Sequence()
HLTIterativeTrackingIter02 = cms.Sequence()
HLTIterativeTracking = cms.Sequence()
HLTIterativeTrackingForHighPt = cms.Sequence()
HLTIterativeTrackingTau3Mu = cms.Sequence()
HLTIterativeTrackingReg = cms.Sequence()
HLTIterativeTrackingForElectronIter02 = cms.Sequence()
HLTIterativeTrackingForPhotonsIter02 = cms.Sequence()
HLTIterativeTrackingL3MuonIter02 = cms.Sequence()
HLTIterativeTrackingL3MuonRegIter02 = cms.Sequence()
HLTIterativeTrackingForBTagIter02 = cms.Sequence()
HLTIterativeTrackingForTauIter02 = cms.Sequence()
HLTIterativeTrackingForTauIter04 = cms.Sequence()
HLTIterativeTrackingGlbTrkMuonIter02 = cms.Sequence()
HLTIterativeTrackingHighPtTkMu = cms.Sequence()
HLTIterativeTrackingHighPtTkMuIsoIter02 = cms.Sequence()
HLTIterativeTrackingDisplacedJpsiIter02     = cms.Sequence()
HLTIterativeTrackingDisplacedPsiPrimeIter02 = cms.Sequence()
HLTIterativeTrackingDisplacedNRMuMuIter02   = cms.Sequence()
HLTIterativeTrackingForBTagIter12 = cms.Sequence()
HLTIterativeTrackingForBTagIteration0 = cms.Sequence()
HLTIterativeTrackingIteration4DisplacedJets = cms.Sequence()
