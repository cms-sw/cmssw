import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import CandVars,Var

trackingParticleTable = cms.EDProducer("SimpleTrackingParticleFlatTableProducer",
    src = cms.InputTag("mix:MergedTrackTruth"),
    cut = cms.string(""), 
    name = cms.string("TrackingPart"),
    doc  = cms.string("TrackingPart"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the muons
    variables = cms.PSet(CandVars,
        nGenPart = Var('genParticles().size()', 'int', precision=-1, doc='Number of associated gen particles'),
        GenPartIdx = Var('? genParticles.size() ? genParticles().at(0).key() : -1', 'int', precision=-1, doc='Number of associated gen particles'),
        trackId = Var('g4Tracks.at(0).trackId', 'int', precision=-1, doc='Geant4 track ID of first track'),
        nSimTrack = Var('g4Tracks().size', 'int', precision=-1, doc='Number of associated simtracks'),
        Vtx_x = Var('vx()', 'float', precision=14, doc='parent vertex x pos'),
        Vtx_y = Var('vy()', 'float', precision=14, doc='parent vertex y pos'),
        Vtx_z = Var('vz()', 'float', precision=14, doc='parent vertex z pos'),
        Vtx_t = Var('parentVertex().position().T()', 'float', precision=14, doc='parent vertex time'),
        nDecayVtx = Var('decayVertices().size()', 'int', precision=-1, doc='number of decay vertices'),
        DecayVtx_y = Var('? decayVertices().size() > 0 ? decayVertices().at(0).position().x : 10000', 'float', precision=14, doc='x position of first decay vertex'),
        DecayVtx_x = Var('? decayVertices().size() > 0 ? decayVertices().at(0).position().y : 10000', 'float', precision=14, doc='y position of first decay vertex'),
        DecayVtx_z = Var('? decayVertices().size() > 0 ? decayVertices().at(0).position().z : 10000', 'float', precision=14, doc='z position of first decay vertex'),
        DecayVtx_t = Var('? decayVertices().size() > 0 ? decayVertices().at(0).position().t : 10000', 'float', precision=14, doc='time of first decay vertex'),
    )
)


