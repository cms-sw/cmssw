import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import Var

simTrackTable = cms.EDProducer("SimpleSimTrackFlatTableProducer",
    src = cms.InputTag("g4SimHits"),
    cut = cms.string("abs(momentum().eta) > 1.52 || abs(getMomentumAtBoundary().eta()) > 1.52"), 
    name = cms.string("SimTrack"),
    doc  = cms.string("Geant4 sim tracks in HGCAL Electromagnetic endcap"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the muons
    variables = cms.PSet(
        crossedBoundary = Var('crossedBoundary', 'bool', doc='track crossed boundary'),
        pdgId = Var('type', 'int', doc='pdgId (track type)'),
        charge = Var('charge', 'int', doc='ID'),
        trackId = Var('trackId', 'int', precision=-1, doc='ID'),
        trackIdAtBoundary = Var('getIDAtBoundary', 'int', precision=-1, doc='ID at boundary crossing'),
        pt = Var('momentum().pt()', 'float', precision=14, doc='pt'),
        eta = Var('momentum().eta()', 'float', precision=14, doc='eta'),
        phi = Var('momentum().phi()', 'float', precision=14, doc='phi'),
        lastPos_x = Var('trackerSurfacePosition().x()', 'float', precision=14, doc='x position at HGCAL boundary'),
        lastPos_y = Var('trackerSurfacePosition().y()', 'float', precision=14, doc='y position at HGCAL boundary'),
        lastPos_z = Var('trackerSurfacePosition().z()', 'float', precision=14, doc='z position at HGCAL boundary'),
        boundaryPos_x = Var('getPositionAtBoundary().x()', 'float', precision=14, doc='x position at HGCAL boundary'),
        boundaryPos_y = Var('getPositionAtBoundary().y()', 'float', precision=14, doc='y position at HGCAL boundary'),
        boundaryPos_z = Var('getPositionAtBoundary().z()', 'float', precision=14, doc='z position at HGCAL boundary'),
        boundaryMomentum_pt = Var('getMomentumAtBoundary().pt()', 'float', precision=14, doc='pt at HGCAL boundary'),
        boundaryMomentum_eta = Var('getMomentumAtBoundary().eta()', 'float', precision=14, doc='eta position at HGCAL boundary'),
        boundaryMomentum_phi = Var('getMomentumAtBoundary().phi()', 'float', precision=14, doc='phi position at HGCAL boundary'),
    )
)

simTrackToSimClusterTable = cms.EDProducer("SimTrackToSimClusterIndexTableProducer",
    cut = simTrackTable.cut,
    src = simTrackTable.src,
    objName = simTrackTable.name,
    branchName = cms.string("SimCluster"),
    objMap = cms.InputTag("mix:simTrackToSimCluster"),
    docString = cms.string("SimCluster containing track")
)

simTrackTables = cms.Sequence(simTrackTable+simTrackToSimClusterTable)
