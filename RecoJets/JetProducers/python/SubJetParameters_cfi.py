import FWCore.ParameterSet.Config as cms

SubJetParameters = cms.PSet(
    nFilt = cms.int32(2),                # number of subjets to decluster the fat jet into (actual value can be less if less than 6 constituents in fat jet).
    zcut = cms.double(0.1),                 # zcut parameter for pruning (see ref. for details)
    rcut_factor = cms.double(0.5)           # rcut factor for pruning (the ref. uses 0.5)
)
