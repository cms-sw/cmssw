import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import CandVars,Var

caloParticleTable = cms.EDProducer("SimpleCaloParticleFlatTableProducer",
    src = cms.InputTag("mix:MergedCaloTruth"),
    cut = cms.string(""), 
    name = cms.string("CaloPart"),
    doc  = cms.string("CaloPart"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the muons
    variables = cms.PSet(CandVars,
        simEnergy = Var('simEnergy', 'int', precision=-1, doc='Number of associated gen particles'),
        nGenPart = Var('genParticles().size()', 'int', precision=-1, doc='Number of associated gen particles'),
        GenPartIdx = Var('? genParticles.size() ? genParticles().at(0).key() : -1', 'int', precision=-1, doc='Number of associated gen particles'),
        nSimHit = Var('numberOfSimHits', 'int', precision=-1, doc='Number of simhits'),
        trackId = Var('g4Tracks().at(0).trackId', 'int', precision=-1, doc='Geant4 track ID of first track'),
        nSimTrack = Var('g4Tracks().size', 'int', precision=-1, doc='Number of associated simtracks'),
    )
)

