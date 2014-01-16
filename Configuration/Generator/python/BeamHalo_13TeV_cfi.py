import FWCore.ParameterSet.Config as cms

generator = cms.EDProducer("BeamHaloProducer",
    GENMOD = cms.untracked.int32(1),
    LHC_B1 = cms.untracked.int32(1),
    LHC_B2 = cms.untracked.int32(1),
    IW_MUO = cms.untracked.int32(1),
    IW_HAD = cms.untracked.int32(0),
    EG_MIN = cms.untracked.double(10.),
    EG_MAX = cms.untracked.double(13000.),
    shift_bx  = cms.untracked.int32(0),   ## e.g. -2, -1 for previous bunch-crossing
    BXNS = cms.untracked.double(25.)      ## time between 2 bx s, in ns
)
