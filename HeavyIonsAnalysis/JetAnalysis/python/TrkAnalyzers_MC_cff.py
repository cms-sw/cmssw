import FWCore.ParameterSet.Config as cms

from HeavyIonsAnalysis.JetAnalysis.TrkAnalyzers_cff import *
from HeavyIonsAnalysis.JetAnalysis.TrkEfficiency_cff import *
from SimTracker.TrackAssociation.TrackAssociatorByChi2_cfi import *
TrackAssociatorByChi2ESProducer.chi2cut = cms.double(10.0)

anaTrack.doSimVertex = True
anaTrack.doSimTrack = True
anaTrack.fillSimTrack = cms.untracked.bool(True)
anaTrack.simTrackPtMin = 0.4

pixelTrack.doSimVertex = True
pixelTrack.doSimTrack = True
pixelTrack.simTrackPtMin = 0.4
pixelTrack.fillSimTrack = cms.untracked.bool(True)

mergedTrack.doSimTrack = True
mergedTrack.simTrackPtMin = 0.4
mergedTrack.fillSimTrack = cms.untracked.bool(True)

anaTrack.tpFakeSrc = cms.untracked.InputTag("cutsTPForFak")
anaTrack.tpEffSrc = cms.untracked.InputTag("cutsTPForEff")
pixelTrack.tpFakeSrc = cms.untracked.InputTag("cutsTPForFak")
pixelTrack.tpEffSrc = cms.untracked.InputTag("cutsTPForEff")
anaTrack.tpFakeSrc = cms.untracked.InputTag("cutsTPForFak")
anaTrack.tpEffSrc = cms.untracked.InputTag("cutsTPForEff")
pixelTrack.tpFakeSrc = cms.untracked.InputTag("cutsTPForFak")
pixelTrack.tpEffSrc = cms.untracked.InputTag("cutsTPForEff")

cutsTPForFak.ptMin = 0.4
cutsTPForEff.ptMin = 0.4

ppTrack.doSimVertex = True
ppTrack.doSimTrack = True
ppTrack.fillSimTrack = cms.untracked.bool(True)
ppTrack.simTrackPtMin = 0.4
ppTrack.tpFakeSrc = cms.untracked.InputTag("cutsTPForFak")
ppTrack.tpEffSrc = cms.untracked.InputTag("cutsTPForEff")
ppTrack.tpFakeSrc = cms.untracked.InputTag("cutsTPForFak")
ppTrack.tpEffSrc = cms.untracked.InputTag("cutsTPForEff")
