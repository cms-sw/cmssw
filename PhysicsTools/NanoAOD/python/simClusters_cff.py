import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import CandVars,Var

simClusterTable = cms.EDProducer("SimpleSimClusterFlatTableProducer",
    src = cms.InputTag("mix:MergedCaloTruth"),
    cut = cms.string(""),
    name = cms.string("SimCluster"),
    doc  = cms.string("SimCluster information"),
    singleton = cms.bool(False), # the number of entries is variable
    extension = cms.bool(False), # this is the main table for the muons
    variables = cms.PSet(CandVars,
        #x = Var('impactPoint().x()', 'float', precision=14, doc='x position'),
        #y = Var('impactPoint().y()', 'float', precision=14, doc='y position'),
        #z = Var('impactPoint().z()', 'float', precision=14, doc='z position'),
        #impactPointX = Var('impactPoint().x()', 'float', precision=14, doc='x position'),
        #impactPointY = Var('impactPoint().y()', 'float', precision=14, doc='y position'),
        #impactPointZ = Var('impactPoint().z()', 'float', precision=14, doc='z position'),
        lastPos_x = Var('g4Tracks.at(0).trackerSurfacePosition().x()', 'float', precision=14, doc='track x final position'),
        lastPos_y = Var('g4Tracks.at(0).trackerSurfacePosition().y()', 'float', precision=14, doc='track y final position'),
        lastPos_z = Var('g4Tracks.at(0).trackerSurfacePosition().z()', 'float', precision=14, doc='track z final position'),
        nSimHits = Var('numberOfSimHits', 'int', precision=-1, doc='total energy of simhits'),
        simEnergy = Var('simEnergy', 'float', precision=14, doc='total energy of simhits'),
        trackId = Var('g4Tracks().at(0).trackId()', 'int', precision=10, doc='Geant track id'),
        #trackIdAtBoundary = Var('g4Tracks().at(0).getIDAtBoundary()', 'int', precision=-1, doc='Track ID at boundary'),
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
