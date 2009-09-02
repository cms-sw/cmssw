import FWCore.ParameterSet.Config as cms

SubJetParameters = cms.PSet(
    #read from CompoundJetProducer:
    subjetColl = cms.string("SubJets"),    # subjet collection name
    #read from SubJetProducer:
    algorithm = cms.int32(1),               # 0 = KT, 1 = CA, 2 = anti-KT
    centralEtaCut = cms.double(2.5),        # eta cut for the "fat jet"
    jetSize = cms.double(0.8),              # jet algorithm cut-off parameter for definition of fat jet (meaning depends on jet algorithm)
    nSubjets = cms.int32(3),                # number of subjets to decluster the fat jet into (actual value can be less if less than 6 constituents in fat jet).
    enable_pruning = cms.bool(True),        # enable pruning (see arXiv:0903.5081)
    zcut = cms.double(0.1),                 # zcut parameter for pruning (see ref. for details)
    rcut_factor = cms.double(0.5)           # rcut factor for pruning (the ref. uses 0.5)
)
