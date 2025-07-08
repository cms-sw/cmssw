import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

from PhysicsTools.EXOnanoAOD.cscMDSshowerTable_cfi import cscMDSshowerTable 
from PhysicsTools.EXOnanoAOD.dtMDSshowerTable_cfi import dtMDSshowerTable 

cscMDSshowerTable = cscMDSshowerTable.clone( 
    name = cms.string("cscMDSCluster"),
    recHitLabel = cms.InputTag("csc2DRecHits"),
    segmentLabel = cms.InputTag("dt4DSegments"),
    rpcLabel = cms.InputTag("rpcRecHits")
)
dtMDSshowerTable = dtMDSshowerTable.clone( 
    name = cms.string("dtMDSCluster"),
    recHitLabel = cms.InputTag("dt1DRecHits"),
    rpcLabel = cms.InputTag("rpcRecHits")
)

#DSA muon tables
DSAmuonsTable = cms.EDProducer("DSAMuonTableProducer",
    dsaMuons=cms.InputTag("displacedStandAloneMuons"),
    muons=cms.InputTag("linkedObjects","muons"),
    primaryVertex = cms.InputTag("offlineSlimmedPrimaryVertices"),
    beamspot = cms.InputTag("offlineBeamSpot")
)

DSAmuonVertexTable = cms.EDProducer("MuonVertexTableProducer",
    dsaMuons=cms.InputTag("displacedStandAloneMuons"),
    patMuons=cms.InputTag("linkedObjects","muons"),
    beamspot=cms.InputTag("offlineBeamSpot"),
    generalTracks=cms.InputTag("generalTracks"),
    primaryVertex=cms.InputTag("offlineSlimmedPrimaryVertices")
)

PATmuonExtendedTable = cms.EDProducer("MuonExtendedTableProducer",
    name=cms.string("Muon"),
    rho=cms.InputTag("fixedGridRhoFastjetAll"),
    muons=cms.InputTag("linkedObjects","muons"),
    dsaMuons=cms.InputTag("displacedStandAloneMuons"),
    primaryVertex=cms.InputTag("offlineSlimmedPrimaryVertices"),
    beamspot=cms.InputTag("offlineBeamSpot"),                        
    generalTracks=cms.InputTag("generalTracks"),
    jets=cms.InputTag("linkedObjects","jets"),
    jetsFat=cms.InputTag("slimmedJetsAK8"),
    jetsSub=cms.InputTag("slimmedJetsAK8PFPuppiSoftDropPacked", "SubJets")
)

electronVertexTable = cms.EDProducer("ElectronVertexTableProducer",
    electrons=cms.InputTag("linkedObjects","electrons"),
    beamspot=cms.InputTag("offlineBeamSpot"),
    generalTracks=cms.InputTag("generalTracks"),
    primaryVertex=cms.InputTag("offlineSlimmedPrimaryVertices")
)

electronExtendedTable = cms.EDProducer("ElectronExtendedTableProducer",
    name=cms.string("Electron"),
    rho=cms.InputTag("fixedGridRhoFastjetAll"),                                       
    electrons=cms.InputTag("linkedObjects","electrons"),
    primaryVertex=cms.InputTag("offlineSlimmedPrimaryVertices"),
    jets=cms.InputTag("linkedObjects","jets"),
    jetsFat=cms.InputTag("slimmedJetsAK8"),
    jetsSub=cms.InputTag("slimmedJetsAK8PFPuppiSoftDropPacked", "SubJets")
)

dispJetTable = cms.EDProducer("DispJetTableProducer",
    electrons=cms.InputTag("linkedObjects","electrons"),
    muons=cms.InputTag("linkedObjects","muons"),
    primaryVertex = cms.InputTag("offlineSlimmedPrimaryVertices"),
    secondaryVertex = cms.InputTag("displacedInclusiveSecondaryVertices")
)

def add_dispJetTables(process):
    process.load('PhysicsTools.EXOnanoAOD.displacedInclusiveVertexing_cff')
    process.dispJetTable = dispJetTable
    process.dispJetTask = cms.Task(process.unpackedTracksAndVertices)
    process.dispJetTask.add(process.displacedInclusiveVertexFinder)
    process.dispJetTask.add(process.displacedVertexMerger)
    process.dispJetTask.add(process.displacedTrackVertexArbitrator)
    process.dispJetTask.add(process.displacedInclusiveSecondaryVertices)
    process.dispJetTask.add(process.dispJetTable)
    process.nanoTableTaskCommon.add(process.dispJetTask)

    return process

def add_mdsTables(process):
    process.cscMDSshowerTable = cscMDSshowerTable    
    process.dtMDSshowerTable = dtMDSshowerTable   

    process.MDSTask = cms.Task(process.cscMDSshowerTable)
    process.MDSTask.add(process.dtMDSshowerTable)

    process.nanoTableTaskCommon.add(process.MDSTask)

    return process

def add_dsamuonTables(process):
    process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
    process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi')

    process.DSAmuonsTable = DSAmuonsTable
    process.DSAmuonVertexTable = DSAmuonVertexTable
    process.PATmuonExtendedTable = PATmuonExtendedTable

    process.dsamuonTask = cms.Task(process.DSAmuonsTable)
    process.dsamuonVertexTask = cms.Task(process.DSAmuonVertexTable)
    process.patmuonTask = cms.Task(process.PATmuonExtendedTable)

    process.nanoTableTaskCommon.add(process.dsamuonTask)
    process.nanoTableTaskCommon.add(process.dsamuonVertexTask)
    process.nanoTableTaskCommon.add(process.patmuonTask)

    return process

def add_electronVertexTables(process):

    process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
    process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi')

    process.electronVertexTable = electronVertexTable
    process.electronVertexTask = cms.Task(process.electronVertexTable)
    process.electronExtendedTable = electronExtendedTable
    process.electronExtendedTask = cms.Task(process.electronExtendedTable)
   
    process.nanoTableTaskCommon.add(process.electronVertexTask)

    return process

def add_muonExtendedTable(process):
    process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
    process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi')

    process.PATmuonExtendedTable = PATmuonExtendedTable

    process.muonExtendedTask = cms.Task(process.PATmuonExtendedTable)

    process.nanoTableTaskCommon.add(process.muonExtendedTask)

    return process

def update_genParticleTable(process):

    process.genParticleTable.variables.vx = Var("vx",float, doc = "gen particle production vertex x coordinate (cm)", precision=8)
    process.genParticleTable.variables.vy = Var("vy",float, doc = "gen particle production vertex y coordinate (cm)", precision=8)
    process.genParticleTable.variables.vz = Var("vz",float, doc = "gen particle production vertex z coordinate (cm)", precision=8)

    process.genParticleTable.variables.px = Var("px",float, doc = "gen particle momentum x coordinate", precision=8)
    process.genParticleTable.variables.py = Var("py",float, doc = "gen particle momentum y coordinate", precision=8)
    process.genParticleTable.variables.pz = Var("pz",float, doc = "gen particle momentum z coordinate", precision=8)

    return process

def add_exonanoTables(process):

    process = add_mdsTables(process)
    process = add_dsamuonTables(process)
    process = add_electronVertexTables(process)
    process = add_dispJetTables(process)

    process = update_genParticleTable(process)

    return process

def add_exonanoTablesMINIAOD(process):

#    process = add_mdsTables(process)
#    process = add_dsamuonTables(process)
    process = add_electronVertexTables(process)
    process = add_muonExtendedTable(process)
    process = add_dispJetTables(process)

    return process

def add_exonanoMCTables(process):

    process = update_genParticleTable(process)

    return process
