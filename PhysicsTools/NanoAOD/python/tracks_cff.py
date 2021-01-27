import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import P3Vars,Var

generalTrackTable = cms.EDProducer("SimpleTrackFlatTableProducer",
    src = cms.InputTag("generalTracks"),
    cut = cms.string(""), 
    name = cms.string("Track"),
    doc  = cms.string("reco::Track"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the muons
    variables = cms.PSet(P3Vars,
        charge = Var("charge", int, doc="electric charge"),
        normChiSq = Var("normalizedChi2", float, precision=14, doc="Chi^2/ndof"),
        numberOfValidHits = Var('numberOfValidHits()', 'int', precision=-1, doc='Number of valid hits in track'),
        numberOfLostHits = Var('numberOfLostHits()', 'int', precision=-1, doc='Number of lost hits in track'),
        Vtx_x = Var('vx()', 'float', precision=14, doc='parent vertex x pos'),
        Vtx_y = Var('vy()', 'float', precision=14, doc='parent vertex y pos'),
        Vtx_z = Var('vz()', 'float', precision=14, doc='parent vertex z pos'),
        Vtx_t = Var('t0', 'float', precision=14, doc='parent vertex time'),
        # Be careful here, this isn't really a decay vertex
        DecayVtx_y = Var('outerPosition().x()', 'float', precision=14, doc='x position of first decay vertex'),
        DecayVtx_x = Var('outerPosition().y()', 'float', precision=14, doc='y position of first decay vertex'),
        DecayVtx_z = Var('outerPosition().z()', 'float', precision=14, doc='z position of first decay vertex'),
        DecayVtx_t = Var('0', 'float', precision=14, doc='DUMMY VALUE! for time of first decay vertex'),
    )
)

trackConversionsTable = generalTrackTable.clone()
trackConversionsTable.src = "conversionStepTracks"
trackConversionsTable.name = "TrackConv"

trackDisplacedTable = cms.EDProducer("SimpleTrackFlatTableProducer",
    src = cms.InputTag("displacedTracks"),
    cut = cms.string(""), 
    name = cms.string("TrackDisp"),
    doc  = cms.string("reco::Track"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the muons
    variables = cms.PSet(P3Vars,
        charge = Var("charge", int, doc="electric charge"),
        Vtx_x = Var('vx()', 'float', precision=14, doc='parent vertex x pos'),
        Vtx_y = Var('vy()', 'float', precision=14, doc='parent vertex y pos'),
        Vtx_z = Var('vz()', 'float', precision=14, doc='parent vertex z pos'),
        Vtx_t = Var('t0', 'float', precision=14, doc='parent vertex time'),
    )
)

trackTables = cms.Sequence(generalTrackTable+trackConversionsTable+trackDisplacedTable)
