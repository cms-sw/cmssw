import FWCore.ParameterSet.Config as cms

# configuration of the BeamProfileFitter
#
BeamProfileFitterBlock = cms.PSet(
    BeamProfileFitter = cms.PSet(
        ScaleHistogramBeforeFit = cms.untracked.bool(True),
        ClearHistogramAfterFit = cms.untracked.bool(True),
        BSAnglesSystematic = cms.untracked.double(0.0007), ## systematic deviation of the measured BS angles w.r.t. the reconstructed ones from the Sector tests

        CorrectBeamSplitterKink = cms.untracked.bool(True), ## set to false for monte carlo events

        MinimalSignalHeight = cms.untracked.double(0.0)
    )
)

