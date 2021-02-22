import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import CandVars,Var

pfCandTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("particleFlow"),
    cut = cms.string(""), 
    name = cms.string("PFCand"),
    doc  = cms.string("ParticleFlow candidates"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the muons
    variables = cms.PSet(CandVars,
        Vtx_x = Var('vertex().x()', 'float', precision=14, doc='vertex x pos'),
        Vtx_y = Var('vertex().y()', 'float', precision=14, doc='vertex y pos'),
        Vtx_z = Var('vertex().z()', 'float', precision=14, doc='vertex z pos'),
    )
)

pfTICLCandTable = cms.EDProducer("SimpleCandidateFlatTableProducer",
    src = cms.InputTag("pfTICL"),
    cut = cms.string(""), 
    name = cms.string("PFTICLCand"),
    doc  = cms.string("ParticleFlow candidates"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), 
    variables = cms.PSet(CandVars,
        Vtx_x = Var('vertex().x()', 'float', precision=14, doc='vertex x pos'),
        Vtx_y = Var('vertex().y()', 'float', precision=14, doc='vertex y pos'),
        Vtx_z = Var('vertex().z()', 'float', precision=14, doc='vertex z pos'),
    )
)

pfCandTables = cms.Sequence(pfCandTable+pfTICLCandTable)
