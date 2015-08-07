import FWCore.ParameterSet.Config as cms

selectedDeDxHarm = {}

from RecoTracker.DeDx.dedxEstimators_cff import dedxHarmonic2
dedxDQMHarm2SP = dedxHarmonic2.clone()
dedxDQMHarm2SP.tracks                     = cms.InputTag("generalTracks")
dedxDQMHarm2SP.trajectoryTrackAssociation = cms.InputTag("generalTracks")
dedxDQMHarm2SP.UseStrip = cms.bool(True)
dedxDQMHarm2SP.UsePixel = cms.bool(True)

dedxDQMHarm2SO = dedxDQMHarm2SP.clone()
dedxDQMHarm2SO.UsePixel = cms.bool(False)

dedxDQMHarm2PO = dedxDQMHarm2SP.clone()
dedxDQMHarm2PO.UseStrip = cms.bool(False)

dedxHarmonicSequence = cms.Sequence()
dedxHarmonicSequence+=dedxDQMHarm2SP
dedxHarmonicSequence+=dedxDQMHarm2SO
dedxHarmonicSequence+=dedxDQMHarm2PO

#dEdxMonitor = cms.Sequence(
#    RefitterForDedxDQMDeDx * dedxDQMHarm2SP * dedxDQMHarm2SO * dedxDQMHarm2PO
#    * dEdxAnalyzer
#)
