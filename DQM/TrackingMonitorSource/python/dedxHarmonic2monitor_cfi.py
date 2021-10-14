import FWCore.ParameterSet.Config as cms

selectedDeDxHarm = {}

from RecoTracker.DeDx.dedxEstimators_cff import dedxHarmonic2
dedxDQMHarm2SP = dedxHarmonic2.clone(
    tracks = "generalTracks",
    UseStrip = True,
    UsePixel = True
)

dedxDQMHarm2SO = dedxDQMHarm2SP.clone(
    UsePixel = False
)

dedxDQMHarm2PO = dedxDQMHarm2SP.clone(
    UseStrip = False
)

dedxHarmonicSequence = cms.Sequence()
dedxHarmonicSequence+=dedxDQMHarm2SP
dedxHarmonicSequence+=dedxDQMHarm2SO
dedxHarmonicSequence+=dedxDQMHarm2PO

#dEdxMonitor = cms.Sequence(
#    RefitterForDedxDQMDeDx * dedxDQMHarm2SP * dedxDQMHarm2SO * dedxDQMHarm2PO
#    * dEdxAnalyzer
#)
