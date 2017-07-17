import FWCore.ParameterSet.Config as cms

AlignmentParameterStore = cms.PSet(
    ParameterStore = cms.PSet(
        ExtendedCorrelationsConfig = cms.PSet(
            CutValue = cms.double(0.95),
            Weight = cms.double(0.5),
            MaxUpdates = cms.int32(5000)
        ),
        UseExtendedCorrelations = cms.untracked.bool(False),
        TypeOfConstraints = cms.string('hierarchy'),
    )
)

