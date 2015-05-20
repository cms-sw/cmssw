import FWCore.ParameterSet.Config as cms

# watch out here, we assume that
#   - loading FastSimulation.Configuration.Digi_cff.py causes generalTracksAliasInfo.key.value() to return "mix"
#   - loading FastSimulation.Configuration.DataMixerPreMix_cff.py causes generalTracksAliasInfo.key.value() to return "dataMix"
# such that the HLT reads the tracks from the right MixingModule

import FastSimulation.Configuration.DigiAndMixAliasInfo_cff as _aliasInfo

hltIter4Merged = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)
hltIter2Merged = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)
hltIter4HighPtMerged = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)
hltIter2HighPtMerged = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)
hltIter4Tau3MuMerged = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)
hltIter4MergedReg = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)
hltIter2MergedForElectrons = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)
hltIter2MergedForPhotons = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)
hltIter2L3MuonMerged = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)
hltIter2L3MuonMergedReg = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)
hltIter2MergedForBTag = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)
hltIter2MergedForTau = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)
hltIter4MergedForTau = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)
hltIter2GlbTrkMuonMerged = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)
hltIter2HighPtTkMuMerged  = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)
hltIter2HighPtTkMuIsoMerged  = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)
hltIter2DisplacedJpsiMerged     = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)
hltIter2DisplacedPsiPrimeMerged = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)
hltIter2DisplacedNRMuMuMerged   = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)
hltIter0PFlowTrackSelectionHighPurityForBTag = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)
hltIter4MergedWithIter012DisplacedJets = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)

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
