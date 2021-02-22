import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import CandVars,Var

simClusterTable = cms.EDProducer("SimpleSimClusterFlatTableProducer",
    src = cms.InputTag("mix:MergedCaloTruth"),
    cut = cms.string(""),
    name = cms.string("SimCluster"),
    doc  = cms.string("SimCluster information"),
    singleton = cms.bool(False),
    extension = cms.bool(False), 
    variables = cms.PSet(CandVars,
        lastPos_x = Var('g4Tracks.at(0).trackerSurfacePosition().x()', 'float', precision=14, doc='track x final position'),
        lastPos_y = Var('g4Tracks.at(0).trackerSurfacePosition().y()', 'float', precision=14, doc='track y final position'),
        lastPos_z = Var('g4Tracks.at(0).trackerSurfacePosition().z()', 'float', precision=14, doc='track z final position'),
        impactPoint_x = Var('g4Tracks().at(0).getPositionAtBoundary().x()', 'float', precision=14, doc='x position'),
        impactPoint_y = Var('g4Tracks().at(0).getPositionAtBoundary().y()', 'float', precision=14, doc='y position'),
        impactPoint_z = Var('g4Tracks().at(0).getPositionAtBoundary().z()', 'float', precision=14, doc='z position'),
        impactPoint_t = Var('g4Tracks().at(0).getPositionAtBoundary().t()', 'float', precision=14, doc='Impact time'),
        # For stupid reasons lost on me, the nsimhits_ variable is uninitialized, and hits_ (which are really simhits)
        # are often referred to as rechits in the SimCluster class
        nHits = Var('numberOfRecHits', 'int', precision=-1, doc='number of simhits'),
        sumHitEnergy = Var('energy', 'float', precision=14, doc='total energy of simhits'),
        boundaryPmag = Var('impactMomentum.P()', 'float', precision=14, doc='magnitude of the boundary 3-momentum vector'),
        boundaryP4 = Var('impactMomentum.mag()', 'float', precision=14, doc='magnitude of four vector'),
        boundaryEnergy = Var('impactMomentum.energy()', 'float', precision=14, doc='magnitude of four vector'),
        boundaryPt = Var('impactMomentum.pt()', 'float', precision=14, doc='magnitude of four vector'),
        trackId = Var('g4Tracks().at(0).trackId()', 'int', precision=-1, doc='Geant track id'),
        trackIdAtBoundary = Var('g4Tracks().at(0).getIDAtBoundary()', 'int', precision=-1, doc='Track ID at boundary'),
    )
)

simClusterToCaloPartTable = cms.EDProducer("SimClusterToCaloParticleIndexTableProducer",
    cut = simClusterTable.cut,
    src = simClusterTable.src,
    objName = simClusterTable.name,
    branchName = cms.string("CaloPart"),
    objMap = cms.InputTag("mix:simClusterToCaloParticle"),
    docString = cms.string("Index of CaloPart containing SimCluster")
)

simClusterTables = cms.Sequence(simClusterTable+simClusterToCaloPartTable)
