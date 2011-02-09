import FWCore.ParameterSet.Config as cms

TUParamsBlock = cms.PSet(
    # Debug flag
    Debug = cms.untracked.bool(False),
    # MiniCrate digi offset in tdc units
    DIGIOFFSET = cms.int32(500),
    # MiniCrate setup time : fine syncronization
    SINCROTIME = cms.int32(0)
)


