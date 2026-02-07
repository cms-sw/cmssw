import FWCore.ParameterSet.Config as cms
from PhysicsTools.NanoAOD.common_cff import *

# displaced tau
from PhysicsTools.NanoAOD.custom_displacedtau_cff import *

unpackedTracksAndVertices = cms.EDProducer('PATTrackAndVertexUnpacker',
                                           slimmedVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
                                           slimmedSecondaryVertices = cms.InputTag("slimmedSecondaryVertices"),
                                           additionalTracks= cms.InputTag("lostTracks"),
                                           packedCandidates = cms.InputTag("packedPFCandidates"))

# ref: https://github.com/cms-sw/cmssw/blob/2bed69b1658e4deeaef914e462741919e9183be3/RecoVertex/AdaptiveVertexFinder/plugins/InclusiveVertexFinder.h#L48

displacedInclusiveVertexFinder  = cms.EDProducer("InclusiveVertexFinder",
                                                 beamSpot = cms.InputTag("offlineBeamSpot"),
                                                 primaryVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
                                                 tracks = cms.InputTag("unpackedTracksAndVertices"),
                                                 minHits = cms.uint32(6), # 8
                                                 maximumLongitudinalImpactParameter = cms.double(20), # 0.3
                                                 minPt = cms.double(0.8), # 0.8
                                                 maxNTracks = cms.uint32(100), # 30
                                                 
                                                 clusterizer = cms.PSet(
                                                     seedMax3DIPSignificance = cms.double(9999.), # 9999.
                                                     seedMax3DIPValue = cms.double(9999.), # 9999.
                                                     seedMin3DIPSignificance = cms.double(1.2), # 1.2
                                                     seedMin3DIPValue = cms.double(0.005), # 0.005
                                                     clusterMaxDistance = cms.double(0.4), # 0.05
                                                     clusterMaxSignificance = cms.double(4.5), # 4.5
                                                     distanceRatio = cms.double(20), # 20
                                                     clusterMinAngleCosine = cms.double(0.5), # 0.5
                                                 ),
                                                     
                                                 vertexMinAngleCosine = cms.double(0.95), # 0.95
                                                 vertexMinDLen2DSig = cms.double(2.5), # 2.5
                                                 vertexMinDLenSig = cms.double(0.5), # 0.5
                                                 fitterSigmacut =  cms.double(3), # 3
                                                 fitterTini = cms.double(256), # 256
                                                 fitterRatio = cms.double(0.25), # 0.25
                                                 useDirectVertexFitter = cms.bool(True), # True
                                                 useVertexReco  = cms.bool(True), # True
                                                 vertexReco = cms.PSet(
                                                     finder = cms.string('avr'),
                                                     primcut = cms.double(1.0), # 1.0
                                                     seccut = cms.double(3), # 3
                                                     smoothing = cms.bool(True)) # True
                                            )
                                            
displacedVertexMerger = cms.EDProducer("VertexMerger",
                                       secondaryVertices = cms.InputTag("displacedInclusiveVertexFinder"),
                                       maxFraction = cms.double(0.7),
                                       minSignificance = cms.double(2))

# ref: https://github.com/cms-sw/cmssw/blob/2bed69b1658e4deeaef914e462741919e9183be3/RecoVertex/AdaptiveVertexFinder/plugins/VertexArbitrators.cc#L54

displacedTrackVertexArbitrator = cms.EDProducer("TrackVertexArbitrator",
                                                beamSpot = cms.InputTag("offlineBeamSpot"),
                                                primaryVertices = cms.InputTag("offlineSlimmedPrimaryVertices"),
                                                tracks = cms.InputTag("unpackedTracksAndVertices"),
                                                secondaryVertices = cms.InputTag("displacedVertexMerger"),
                                                dLenFraction = cms.double(0.333), # 0.333
                                                dRCut = cms.double(1), # 0.4
                                                distCut = cms.double(0.1), # 0.04
                                                sigCut = cms.double(5), # 5
                                                fitterSigmacut =  cms.double(3), # 3
                                                fitterTini = cms.double(256), # 256
                                                fitterRatio = cms.double(0.25), # 0.25
                                                trackMinLayers = cms.int32(4), # 4
                                                trackMinPt = cms.double(.4), # 0.4
                                                trackMinPixels = cms.int32(0) # 1
)
    
displacedInclusiveSecondaryVertices = displacedVertexMerger.clone()
displacedInclusiveSecondaryVertices.secondaryVertices = cms.InputTag("displacedTrackVertexArbitrator")
displacedInclusiveSecondaryVertices.maxFraction = 0.2
displacedInclusiveSecondaryVertices.minSignificance = 10
    
displacedInclusiveVertexing = cms.Sequence(unpackedTracksAndVertices * displacedInclusiveVertexFinder  * displacedVertexMerger * displacedTrackVertexArbitrator * displacedInclusiveSecondaryVertices)

# MDSnano tables 
from RecoMuon.MuonRechitClusterProducer.cscRechitClusterProducer_cfi import cscRechitClusterProducer
from RecoMuon.MuonRechitClusterProducer.dtRechitClusterProducer_cfi import dtRechitClusterProducer 

from PhysicsTools.NanoAOD.cscMDSshowerTable_cfi import cscMDSshowerTable
from PhysicsTools.NanoAOD.dtMDSshowerTable_cfi import dtMDSshowerTable

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
from PhysicsTools.NanoAOD.simpleTrackFlatTableProducer_cfi import simpleTrackFlatTableProducer
DSAmuonsSimpleTable = simpleTrackFlatTableProducer.clone(
    src="displacedStandAloneMuons",
    name="DSAMuon",
    doc ="Displaced standalone muon tracks",
    singleton=False,
    extension=False,
    variables=cms.PSet(
        pt = Var("pt", float, doc="pt of DSA muon"),
        ptErr = Var("ptError", float, doc="ptErr of DSA muon"),
        eta = Var("eta", float, doc="eta of DSA muon"),
        etaErr = Var("etaError", float, doc="etaErr of DSA muon"),
        phi = Var("phi", float, doc="phi of DSA muon"),
        phiErr = Var("phiError", float, doc="phiErr of DSA muon"),
        charge = Var("charge", float, doc="charge of DSA muon"),
        dxy = Var("dxy", float, doc="dxy of DSA muon"),
        dz = Var("dz", float, doc="dz of DSA muon"),
        vx = Var("vx", float, doc="vx of DSA muon"),
        vy = Var("vy", float, doc="vy of DSA muon"),
        vz = Var("vz", float, doc="vz of DSA muon"),
        chi2 = Var("chi2", float, doc="chi2 of DSA muon"),
        ndof = Var("ndof", float, doc="ndof of DSA muon"),
        normChi2 = Var("normalizedChi2", float, doc="normChi2 of DSA muon"),
    )
)

DSAmuonsTable = cms.EDProducer("DSAMuonTableProducer",
    name=cms.string("DSAMuon"),
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

from PhysicsTools.NanoAOD.simplePATMuonFlatTableProducer_cfi import simplePATMuonFlatTableProducer
PATmuonExtendedSimpleTable = simplePATMuonFlatTableProducer.clone(
    src=cms.InputTag("linkedObjects","muons"),
    name="Muon",
    singleton=False,
    extension=True,
    variables=cms.PSet(
        innerTrackValidFraction = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().validFraction() : -1", float, doc=""),
        globalTrackNormalizedChi2 = Var("? globalTrack().isNonnull() ? globalTrack().normalizedChi2() : -1", float, doc=""),
        CQChi2Position = Var("combinedQuality().chi2LocalPosition", float, doc=""),
        CQTrackKink = Var("combinedQuality().trkKink", float, doc=""),
        numberOfMatchedStation = Var("numberOfMatchedStations", float, doc=""),
        numberOfValidPixelHits = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().hitPattern().numberOfValidPixelHits() : 0", float, doc=""),
        numberOfValidTrackerHits = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().hitPattern().numberOfValidTrackerHits() : 0", float, doc=""),
        trackerLayersWithMeasurement = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().hitPattern().trackerLayersWithMeasurement() : 0", float, doc=""),
        numberInnerHits = Var("? globalTrack().isNonnull() ? globalTrack().hitPattern().numberOfValidMuonHits() : ? outerTrack().isNonnull() ? outerTrack().hitPattern().numberOfValidMuonHits() : 0", float, doc=""),
        innerVx = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().vx() : -1", float, doc=""),
        innerVy = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().vy() : -1", float, doc=""),
        innerVz = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().vz() : -1", float, doc=""),
        innerPt = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().pt() : -1", float, doc=""),
        innerEta = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().eta() : -5", float, doc=""),
        innerPhi = Var("? innerTrack().isNonnull() && innerTrack().isAvailable() ? innerTrack().phi() : -5", float, doc=""),
    )
)

PATmuonExtendedTable = cms.EDProducer("MuonExtendedTableProducer",
    name=cms.string("Muon"),
    rho=cms.InputTag("fixedGridRhoFastjetAll"),
    muons=cms.InputTag("linkedObjects","muons"),
    primaryVertex=cms.InputTag("offlineSlimmedPrimaryVertices"),
    beamspot=cms.InputTag("offlineBeamSpot"),                        
    jets=cms.InputTag("linkedObjects","jets"),
    jetsFat=cms.InputTag("slimmedJetsAK8"),
    jetsSub=cms.InputTag("slimmedJetsAK8PFPuppiSoftDropPacked", "SubJets")
)

electronVertexTable = cms.EDProducer("ElectronVertexTableProducer",
    electrons=cms.InputTag("linkedObjects","electrons"),
    beamspot=cms.InputTag("offlineBeamSpot"),
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
    # process.load('PhysicsTools.displacedInclusiveVertexing_cff')
    process.dispJetTable = dispJetTable
    process.unpackedTracksAndVertices = unpackedTracksAndVertices
    process.displacedInclusiveVertexFinder = displacedInclusiveVertexFinder
    process.displacedVertexMerger = displacedVertexMerger
    process.displacedTrackVertexArbitrator = displacedTrackVertexArbitrator
    process.displacedInclusiveSecondaryVertices = displacedInclusiveSecondaryVertices
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
    process.ca4CSCrechitClusters = cscRechitClusterProducer    
    process.ca4DTrechitClusters = dtRechitClusterProducer
    process.cscMDSshowerTable = cscMDSshowerTable
    process.dtMDSshowerTable = dtMDSshowerTable

    process.MDSTask = cms.Task(process.ca4CSCrechitClusters)
    process.MDSTask.add(process.ca4DTrechitClusters)
    process.MDSTask.add(process.cscMDSshowerTable)
    process.MDSTask.add(process.dtMDSshowerTable)

    process.nanoTableTaskCommon.add(process.MDSTask)

    return process

def add_dsamuonTables(process):
    process.load("TrackingTools.TransientTrack.TransientTrackBuilder_cfi")
    process.load('TrackPropagation.SteppingHelixPropagator.SteppingHelixPropagatorAny_cfi')

    process.DSAmuonsSimpleTable = DSAmuonsSimpleTable
    process.DSAmuonsTable = DSAmuonsTable
    process.DSAmuonVertexTable = DSAmuonVertexTable
    process.PATmuonExtendedSimpleTable = PATmuonExtendedSimpleTable
    process.PATmuonExtendedTable = PATmuonExtendedTable
    
    process.dsamuonTask = cms.Task(
        process.DSAmuonsSimpleTable,
        process.DSAmuonsTable
    )
    process.dsamuonVertexTask = cms.Task(process.DSAmuonVertexTable)
    process.patmuonTask = cms.Task(
        process.PATmuonExtendedSimpleTable,
        process.PATmuonExtendedTable
    )

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

    isMC = hasattr(process, "nanoSequenceMC") and process.schedule.contains(process.nanoSequenceMC)

    process = add_displacedtauCHSTables(process, isMC)

    if isMC:
        process = update_genParticleTable(process)

    return process
