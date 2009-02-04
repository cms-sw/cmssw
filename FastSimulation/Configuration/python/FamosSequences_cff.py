import FWCore.ParameterSet.Config as cms

# Primary vertex smearing.
from IOMC.EventVertexGenerators.VtxSmearedGauss_cfi import *

# Conversion to GenParticleCandidates 
from PhysicsTools.HepMCCandAlgos.genParticleCandidatesFast_cfi import *

# Famos PileUp Producer
from FastSimulation.PileUpProducer.PileUpProducer_cff import *

# Famos SimHits producer
from FastSimulation.EventProducer.FamosSimHits_cff import *

# Mixing module 
from FastSimulation.Configuration.mixNoPU_cfi import *

# Gaussian Smearing RecHit producer
from FastSimulation.TrackingRecHitProducer.SiTrackerGaussianSmearingRecHitConverter_cfi import *

# Rec Hit Tranlator to the Full map with DeTId'
from FastSimulation.TrackingRecHitProducer.TrackingRecHitTranslator_cfi import *

# CTF and Iterative tracking (contains pixelTracks and pixelVertices)

# 1) Common algorithms and configuration taken from full reconstruction
# Note: The runge-kutta propagator is not used here 
# (because no magnetic field inhomogeneities are simulated between layers)
from FastSimulation.Tracking.GSTrackFinalFitCommon_cff import *

# 2) Specific cuts - not needed anymore, as a specific KFFittingSmoother deals with that.
# Add a chi**2 cut to retain/reject hits
# KFFittingSmoother.EstimateCut = 15.0
# Request three hits to make a track
# KFFittingSmoother.MinNumberOfHits = 3

# 3) Fast Simulation tracking sequences
from FastSimulation.Tracking.GlobalPixelTracking_cff import *
from FastSimulation.Tracking.IterativeTracking_cff import *

# Calo RecHits producer (with no HCAL miscalibration by default)
from FastSimulation.CaloRecHitsProducer.CaloRecHits_cff import *

# ECAL clusters
from RecoEcal.Configuration.RecoEcal_cff import *
hybridSuperClusters.ecalhitproducer = 'caloRecHits'
correctedHybridSuperClusters.recHitProducer = cms.InputTag("caloRecHits","EcalRecHitsEB")
correctedMulti5x5SuperClustersWithPreshower.preshRecHitProducer = cms.InputTag("caloRecHits","EcalRecHitsES")
reducedEcalRecHitsEB.recHitsLabel = cms.InputTag("caloRecHits","EcalRecHitsEB")
reducedEcalRecHitsEE.recHitsLabel = cms.InputTag("caloRecHits","EcalRecHitsEE")
interestingEcalDetIdEB.recHitsLabel = cms.InputTag("caloRecHits","EcalRecHitsEB")
interestingEcalDetIdEE.recHitsLabel = cms.InputTag("caloRecHits","EcalRecHitsEE")
multi5x5BasicClusters.barrelHitProducer = 'caloRecHits'
multi5x5BasicClusters.endcapHitProducer = 'caloRecHits'
multi5x5PreshowerClusterShape.preshRecHitProducer = cms.InputTag("caloRecHits","EcalRecHitsES")
multi5x5SuperClustersWithPreshower.preshRecHitProducer = cms.InputTag("caloRecHits","EcalRecHitsES")
correctedMulti5x5SuperClustersWithPreshower.recHitProducer = cms.InputTag("caloRecHits","EcalRecHitsEE")

# Calo Towers
from RecoJets.Configuration.CaloTowersRec_cff import *
towerMaker.ecalInputs = cms.VInputTag(
    cms.InputTag("caloRecHits","EcalRecHitsEB"),
    cms.InputTag("caloRecHits","EcalRecHitsEE")
)
towerMaker.hbheInput = 'caloRecHits'
towerMaker.hfInput = 'caloRecHits'
towerMaker.hoInput = 'caloRecHits'

towerMakerWithHO.ecalInputs = cms.VInputTag(
    cms.InputTag("caloRecHits","EcalRecHitsEB"),
    cms.InputTag("caloRecHits","EcalRecHitsEE")
)
towerMakerWithHO.hbheInput = 'caloRecHits'
towerMakerWithHO.hfInput = 'caloRecHits'
towerMakerWithHO.hoInput = 'caloRecHits'


# Particle Flow
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import *
from RecoParticleFlow.PFTracking.particleFlowTrack_cff import *
from RecoParticleFlow.PFBlockProducer.particleFlowSimParticle_cff import *
from RecoParticleFlow.PFBlockProducer.particleFlowBlock_cff import *
from RecoParticleFlow.PFProducer.particleFlow_cff import *

particleFlowRecHitECAL.ecalRecHitsEB = cms.InputTag("caloRecHits","EcalRecHitsEB")
particleFlowRecHitECAL.ecalRecHitsEE = cms.InputTag("caloRecHits","EcalRecHitsEE")
particleFlowRecHitPS.ecalRecHitsES = cms.InputTag("caloRecHits","EcalRecHitsES")
particleFlowSimParticle.sim = 'famosSimHits'
elecpreid.NHitsInSeed = 1

import FastSimulation.Tracking.TrackCandidateProducer_cfi
fsGsfElCandidates = FastSimulation.Tracking.TrackCandidateProducer_cfi.trackCandidateProducer.clone()
fsGsfElCandidates.SeedProducer = cms.InputTag("elecpreid","SeedsForGsf")
fsGsfElCandidates.TrackProducers = []
fsGsfElCandidates.MinNumberOfCrossedLayers = 5

import TrackingTools.GsfTracking.GsfElectronFit_cfi
fsgsfPFtracks = TrackingTools.GsfTracking.GsfElectronFit_cfi.GsfGlobalElectronTest.clone()
fsgsfPFtracks.src = 'fsGsfElCandidates'
fsgsfPFtracks.TTRHBuilder = 'WithoutRefit'
fsgsfPFtracks.TrajectoryInEvent = True

pfTrackElec.GsfTrackModuleLabel = 'fsgsfPFtracks'

famosParticleFlowSequence = cms.Sequence(
    caloTowersRec+
    particleFlowCluster+
    elecpreid+
    fsGsfElCandidates+
    fsgsfPFtracks+
    pfTrackElec+
    particleFlowBlock+
    particleFlow
)

# Reco Jets and MET
from RecoJets.Configuration.RecoJets_cff import *
from RecoJets.Configuration.RecoPFJets_cff import *
from RecoMET.Configuration.RecoMET_cff import *
from RecoMET.Configuration.RecoPFMET_cff import *
calotoweroptmaker.hbheInput = 'caloRecHits'
calotoweroptmaker.hoInput = 'caloRecHits'
calotoweroptmaker.hfInput = 'caloRecHits'
calotoweroptmaker.ecalInputs = cms.VInputTag(
    cms.InputTag("caloRecHits","EcalRecHitsEB"),
    cms.InputTag("caloRecHits","EcalRecHitsEE")
)

calotoweroptmakerWithHO.hbheInput = 'caloRecHits'
calotoweroptmakerWithHO.hoInput = 'caloRecHits'
calotoweroptmakerWithHO.hfInput = 'caloRecHits'
calotoweroptmakerWithHO.ecalInputs = cms.VInputTag(
    cms.InputTag("caloRecHits","EcalRecHitsEB"),
    cms.InputTag("caloRecHits","EcalRecHitsEE")
)

caloJetMet = cms.Sequence(
    recoJets+
    metreco
)

PFJetMet = cms.Sequence(
    recoPFJets+
    recoPFMET
)

# Gen Jets
from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.Configuration.RecoGenJets_cff import *
from RecoMET.Configuration.GenMETParticles_cff import *
from RecoMET.Configuration.RecoGenMET_cff import *
# No longer applicable according to Ronny
#genCandidatesForMET.verbose = False
caloJetMetGen = cms.Sequence(
    genParticles+
    genJetParticles+
    recoGenJets+
    genMETParticles+
    recoGenMET
)

# Muon parametrization
from FastSimulation.ParamL3MuonProducer.ParamL3Muon_cfi import *

# Muon simHit sequence 
from FastSimulation.MuonSimHitProducer.MuonSimHitProducer_cfi import *

# Muon Digi sequence
from SimMuon.Configuration.SimMuon_cff import *
simMuonCSCDigis.strips.doCorrelatedNoise = False ## Saves a little bit of time

simMuonCSCDigis.InputCollection = 'MuonSimHitsMuonCSCHits'
simMuonDTDigis.InputCollection = 'MuonSimHitsMuonDTHits'
simMuonRPCDigis.InputCollection = 'MuonSimHitsMuonRPCHits'

# Muon RecHit sequence
from RecoLocalMuon.Configuration.RecoLocalMuon_cff import *
csc2DRecHits.stripDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCStripDigi")
csc2DRecHits.wireDigiTag = cms.InputTag("simMuonCSCDigis","MuonCSCWireDigi")
rpcRecHits.rpcDigiLabel = 'simMuonRPCDigis'
dt1DRecHits.dtDigiLabel = 'simMuonDTDigis'

# Muon reconstruction sequence
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *
KFSmootherForMuonTrackLoader.Propagator = 'SmartPropagatorAny'
from RecoMuon.MuonSeedGenerator.standAloneMuonSeeds_cff import *
from RecoMuon.StandAloneMuonProducer.standAloneMuons_cff import *
from FastSimulation.Configuration.globalMuons_cff import *
globalMuons.GLBTrajBuilderParameters.TrackRecHitBuilder = 'WithoutRefit'
globalMuons.GLBTrajBuilderParameters.TrackTransformer.TrackerRecHitBuilder = 'WithoutRefit'
globalMuons.GLBTrajBuilderParameters.TrackerRecHitBuilder = 'WithoutRefit'
globalMuons.GLBTrajBuilderParameters.TransformerOutPropagator = cms.string('SmartPropagatorAny')
globalMuons.GLBTrajBuilderParameters.MatcherOutPropagator = cms.string('SmartPropagator')

from RecoMuon.GlobalMuonProducer.tevMuons_cfi import *
GlobalMuonRefitter.TrackerRecHitBuilder = 'WithoutRefit'
GlobalMuonRefitter.Propagator = 'SmartPropagatorAny'
GlobalTrajectoryBuilderCommon.TrackerRecHitBuilder = 'WithoutRefit'
tevMuons.RefitterParameters.TrackerRecHitBuilder = 'WithoutRefit'
tevMuons.RefitterParameters.Propagator =  'SmartPropagatorAny'
KFSmootherForRefitInsideOut.Propagator = 'SmartPropagatorAny'
KFSmootherForRefitOutsideIn.Propagator = 'SmartPropagator'
KFFitterForRefitInsideOut.Propagator = 'SmartPropagatorAny'
KFFitterForRefitOutsideIn.Propagator = 'SmartPropagatorAny'

famosMuonSequence = cms.Sequence(
    muonDigi+
    muonlocalreco+
    ancientMuonSeed+
    standAloneMuons+
    globalMuons+
    tevMuons
)

#Muon identification sequence
from FastSimulation.Configuration.muonIdentification_cff import *
# Use FastSim tracks and calo hits for muon id
muons.inputCollectionLabels = cms.VInputTag(
    'generalTracks',
    'globalMuons',
    cms.InputTag("standAloneMuons","UpdatedAtVtx")
)
muons.TrackAssociatorParameters.EBRecHitCollectionLabel = cms.InputTag("caloRecHits","EcalRecHitsEB")
muons.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("caloRecHits","EcalRecHitsEE")
muons.TrackAssociatorParameters.CaloTowerCollectionLabel = 'towerMaker'
muons.TrackAssociatorParameters.HBHERecHitCollectionLabel = 'caloRecHits'
muons.TrackAssociatorParameters.HORecHitCollectionLabel = 'caloRecHits'
# Use FastSim tracks and calo hits for calo muon id
calomuons.inputTracks = 'generalTracks'
calomuons.TrackAssociatorParameters.EBRecHitCollectionLabel = cms.InputTag("caloRecHits","EcalRecHitsEB")
calomuons.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("caloRecHits","EcalRecHitsEE")
calomuons.TrackAssociatorParameters.CaloTowerCollectionLabel = 'towerMaker'
calomuons.TrackAssociatorParameters.HBHERecHitCollectionLabel = 'caloRecHits'
calomuons.TrackAssociatorParameters.HORecHitCollectionLabel = 'caloRecHits'

# Muon isolation
from RecoMuon.MuonIsolationProducers.muIsolation_cff import *

famosMuonIdAndIsolationSequence = cms.Sequence(
    sisCone5CaloJets+
    muonIdProducerSequence+
    muIsolation
)

# Electron reconstruction
from FastSimulation.Tracking.globalCombinedSeeds_cfi import *
from FastSimulation.EgammaElectronAlgos.fastElectronSeeds_cfi import *
from FastSimulation.EgammaElectronAlgos.electronGSGsfTrackCandidates_cff import *
from RecoEgamma.EgammaElectronProducers.pixelMatchGsfElectrons_cff import *
from TrackingTools.GsfTracking.GsfElectronFit_cff import *
import TrackingTools.GsfTracking.GsfElectronFit_cfi

pixelMatchGsfFit = TrackingTools.GsfTracking.GsfElectronFit_cfi.GsfGlobalElectronTest.clone()
pixelMatchGsfFit.src = 'electronGSGsfTrackCandidates'
pixelMatchGsfFit.TTRHBuilder = 'WithoutRefit'
pixelMatchGsfFit.TrajectoryInEvent = True
pixelMatchGsfElectrons.barrelSuperClusters = cms.InputTag("correctedHybridSuperClusters","fastElectronSeeds")
pixelMatchGsfElectrons.endcapSuperClusters = cms.InputTag("correctedEndcapSuperClustersWithPreshower","fastElectronSeeds")
pixelMatchGsfElectrons.reducedEndcapRecHitCollection = cms.InputTag("caloRecHits","EcalRecHitsEE")
pixelMatchGsfElectrons.reducedBarrelRecHitCollection = cms.InputTag("caloRecHits","EcalRecHitsEB")

from RecoEgamma.ElectronIdentification.electronIdSequence_cff import *
eidRobustTight.reducedBarrelRecHitCollection = cms.InputTag("caloRecHits","EcalRecHitsEB")
eidRobustTight.reducedEndcapRecHitCollection = cms.InputTag("caloRecHits","EcalRecHitsEE")
eidRobustHighEnergy.reducedBarrelRecHitCollection = cms.InputTag("caloRecHits","EcalRecHitsEB")
eidRobustHighEnergy.reducedEndcapRecHitCollection = cms.InputTag("caloRecHits","EcalRecHitsEE")
eidRobustLoose.reducedBarrelRecHitCollection = cms.InputTag("caloRecHits","EcalRecHitsEB")
eidRobustLoose.reducedEndcapRecHitCollection = cms.InputTag("caloRecHits","EcalRecHitsEE")
eidTight.reducedBarrelRecHitCollection = cms.InputTag("caloRecHits","EcalRecHitsEB")
eidTight.reducedEndcapRecHitCollection = cms.InputTag("caloRecHits","EcalRecHitsEE")
eidLoose.reducedBarrelRecHitCollection = cms.InputTag("caloRecHits","EcalRecHitsEB")
eidLoose.reducedEndcapRecHitCollection = cms.InputTag("caloRecHits","EcalRecHitsEE")
famosElectronSequence = cms.Sequence(
    iterativeFirstSeeds+
    newCombinedSeeds+
    fastElectronSeeds+
    electronGSGsfTrackCandidates+
    pixelMatchGsfFit+
    pixelMatchGsfElectrons+
    eIdSequence
)

# Photon reconstruction
from RecoEgamma.EgammaPhotonProducers.photonSequence_cff import *
photons.barrelEcalHits = cms.InputTag("caloRecHits","EcalRecHitsEB")
photons.endcapEcalHits = cms.InputTag("caloRecHits","EcalRecHitsEE")
photons.isolationSumsCalculatorSet.barrelEcalRecHitProducer = 'caloRecHits'
photons.isolationSumsCalculatorSet.endcapEcalRecHitProducer = 'caloRecHits'
photons.hbheModule = 'caloRecHits'
photons.hbheInstance = ''
photons.pixelSeedProducer = 'fastElectronSeeds'
from RecoEgamma.PhotonIdentification.photonId_cff import *

famosPhotonSequence = cms.Sequence(
    photonSequence+
    photonIDSequence
)

# Add pre-calculated isolation sums for electrons (NB for photons they are stored in the Photon. All is done in the
# sequence above
from RecoEgamma.EgammaIsolationAlgos.egammaIsolationSequence_cff import *
electronEcalRecHitIsolationLcone.ecalBarrelRecHitProducer =  cms.InputTag("caloRecHits")
electronEcalRecHitIsolationLcone.ecalBarrelRecHitCollection =  cms.InputTag("EcalRecHitsEB")

electronEcalRecHitIsolationLcone.ecalEndcapRecHitProducer =  cms.InputTag("caloRecHits")
electronEcalRecHitIsolationLcone.ecalEndcapRecHitCollection =  cms.InputTag("EcalRecHitsEE")

electronEcalRecHitIsolationScone.ecalBarrelRecHitProducer =  cms.InputTag("caloRecHits")
electronEcalRecHitIsolationScone.ecalBarrelRecHitCollection =  cms.InputTag("EcalRecHitsEB")

electronEcalRecHitIsolationScone.ecalEndcapRecHitProducer =  cms.InputTag("caloRecHits")
electronEcalRecHitIsolationScone.ecalEndcapRecHitCollection =  cms.InputTag("EcalRecHitsEE")


#Add egamma ecal interesting rec hits
from RecoEgamma.EgammaIsolationAlgos.interestingEleIsoDetIdModule_cff import *
interestingEleIsoDetId.recHitsLabel = cms.InputTag("caloRecHits","EcalRecHitsEB")
from RecoEgamma.EgammaIsolationAlgos.interestingGamIsoDetIdModule_cff import *
interestingGamIsoDetId.recHitsLabel = cms.InputTag("caloRecHits","EcalRecHitsEB")

from RecoEgamma.EgammaIsolationAlgos.interestingEgammaIsoDetIdsSequence_cff import *
#import  RecoEgamma.EgammaIsolationAlgos.interestingEgammaIsoDetIdsSequence_cff
interestingEleIsoDetIdEB.recHitsLabel = cms.InputTag("caloRecHits","EcalRecHitsEB")
interestingEleIsoDetIdEE.recHitsLabel = cms.InputTag("caloRecHits","EcalRecHitsEE")
interestingGamIsoDetIdEB.recHitsLabel = cms.InputTag("caloRecHits","EcalRecHitsEB")
interestingGamIsoDetIdEE.recHitsLabel = cms.InputTag("caloRecHits","EcalRecHitsEB")




# B tagging
from RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi import *
from RecoJets.JetAssociationProducers.ic5PFJetTracksAssociatorAtVertex_cfi import *
ic5JetTracksAssociatorAtVertex.tracks = 'generalTracks'
ic5PFJetTracksAssociatorAtVertex.tracks = 'generalTracks'
from RecoVertex.Configuration.RecoVertex_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoBTag.Configuration.RecoBTag_cff import *
offlinePrimaryVerticesWithBS.TrackLabel = 'generalTracks'
btagSoftElectrons.HBHERecHitTag = 'caloRecHits'
btagSoftElectrons.TrackAssociatorParameters.EBRecHitCollectionLabel = cms.InputTag("caloRecHits","EcalRecHitsEB")
btagSoftElectrons.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("caloRecHits","EcalRecHitsEE")
btagSoftElectrons.TrackAssociatorParameters.CaloTowerCollectionLabel = 'towerMaker'
btagSoftElectrons.TrackAssociatorParameters.HBHERecHitCollectionLabel = 'caloRecHits'

famosBTaggingSequence = cms.Sequence(
    btagging
)

#Tau tagging
from RecoTauTag.Configuration.RecoTauTag_cff import *
caloRecoTauTagInfoProducer.EBRecHitsSource = cms.InputTag("caloRecHits","EcalRecHitsEB")
caloRecoTauTagInfoProducer.EERecHitsSource = cms.InputTag("caloRecHits","EcalRecHitsEE")
caloRecoTauProducer.EBRecHitsSource = cms.InputTag("caloRecHits","EcalRecHitsEB")
caloRecoTauProducer.EERecHitsSource = cms.InputTag("caloRecHits","EcalRecHitsEE")
caloRecoTauProducer.ESRecHitsSource = cms.InputTag("caloRecHits","EcalRecHitsES")

famosTauTaggingSequence = cms.Sequence(tautagging)

from RecoTauTag.Configuration.RecoPFTauTag_cff import *

famosPFTauTaggingSequence = cms.Sequence(PFTau)

# The sole simulation sequence
famosSimulationSequence = cms.Sequence(
    offlineBeamSpot+
    famosPileUp+
    famosSimHits+
    MuonSimHits+
    mix
)

# Famos pre-defined sequences (and self-explanatory names)
famosWithTrackerHits = cms.Sequence(
    famosSimulationSequence+
    siTrackerGaussianSmearingRecHits
)

famosWithTrackerAndCaloHits = cms.Sequence(
    famosWithTrackerHits+
    caloRecHits
)

famosWithTracks = cms.Sequence(
    famosWithTrackerHits+
    iterativeTracking
)

famosWithTracksAndMuonHits = cms.Sequence(
    famosSimulationSequence+
    siTrackerGaussianSmearingRecHits+
    iterativeTracking+
    famosMuonSequence
)

famosWithTracksAndMuons = cms.Sequence(
    famosSimulationSequence+
    siTrackerGaussianSmearingRecHits+
    iterativeTracking+
    famosMuonSequence+
    caloRecHits+
    caloTowersRec+
    famosMuonIdAndIsolationSequence
)

famosWithCaloHits = cms.Sequence(
    famosSimulationSequence+
    caloRecHits
)

famosWithEcalClusters = cms.Sequence(
    famosWithCaloHits+
    ecalClusters
)

famosWithTracksAndCaloHits = cms.Sequence(
    famosWithTracks+
    caloRecHits
)

famosWithTracksAndEcalClusters = cms.Sequence(
    famosWithTracksAndCaloHits+
    ecalClusters
)

famosWithParticleFlow = cms.Sequence(
    famosWithTracksAndEcalClusters+
    vertexreco+
    caloTowersRec+ 
    famosElectronSequence+
    famosParticleFlowSequence+
    PFJetMet
)

famosWithCaloTowers = cms.Sequence(
    famosWithCaloHits+
    caloTowersRec
)

famosWithJets = cms.Sequence(
    famosWithCaloTowers+
    caloJetMetGen+
    caloJetMet
)

famosWithTracksAndCaloTowers = cms.Sequence(
    famosWithTracksAndCaloHits+
    caloTowersRec
)

famosWithTracksAndJets = cms.Sequence(
    famosWithTracksAndCaloTowers+
    caloJetMetGen+
    caloJetMet
)

famosWithCaloTowersAndParticleFlow = cms.Sequence(
    famosWithParticleFlow+
    caloTowersRec
)

famosWithMuons = cms.Sequence(
    famosWithTracks+
    paramMuons
)

famosWithMuonsAndIsolation = cms.Sequence(
    famosWithTracksAndCaloTowers+
    paramMuons+
    sisCone5CaloJets+
    muIsolation_ParamGlobalMuons
)

famosWithElectrons = cms.Sequence(
    famosWithTracks+
    caloRecHits+
    ecalClusters+ 
    famosElectronSequence+
    interestingEleIsoDetIdEB+
    interestingEleIsoDetIdEE+
    egammaIsolationSequence
)

famosWithPhotons = cms.Sequence(
    famosWithTracks+
    vertexreco+
    caloRecHits+
    ecalClusters+
    famosPhotonSequence+
    interestingGamIsoDetIdEB+
    interestingGamIsoDetIdEE
)

famosWithElectronsAndPhotons = cms.Sequence(
    famosWithTracks+
    vertexreco+
    caloRecHits+
    ecalClusters+
    famosElectronSequence+
    famosPhotonSequence+
    interestingEgammaIsoDetIds+
    egammaIsolationSequence
)

famosWithBTagging = cms.Sequence(
    famosWithTracksAndCaloTowers+
    vertexreco+
    iterativeCone5CaloJets+
    ic5JetTracksAssociatorAtVertex+
    ecalClusters+
    famosMuonSequence+
    reducedRecHitsSequence+ 
    famosBTaggingSequence
    )

famosWithTauTagging = cms.Sequence(
    famosWithTracksAndCaloTowers+
    vertexreco+
    iterativeCone5CaloJets+
    ic5JetTracksAssociatorAtVertex+
    ecalClusters+
    famosTauTaggingSequence
)

famosWithPFTauTagging = cms.Sequence(
    famosWithCaloTowersAndParticleFlow+
    famosPFTauTaggingSequence
)

# The simulation sequence
simulationWithFamos = cms.Sequence(
    famosSimulationSequence+
    siTrackerGaussianSmearingRecHits+
    caloRecHits
)

# The reconstruction sequence
reconstructionWithFamos = cms.Sequence(
    iterativeTracking+
    vertexreco+
    caloTowersRec+
    ecalClusters+
    famosElectronSequence+
    famosPhotonSequence+
    egammaIsolationSequence+
    interestingEgammaIsoDetIds+
    famosMuonSequence+
    famosMuonIdAndIsolationSequence+
    famosParticleFlowSequence+
    caloJetMetGen+
    caloJetMet+
    PFJetMet+
#    paramMuons+
#    muIsolation_ParamGlobalMuons+
    ic5JetTracksAssociatorAtVertex+
    famosTauTaggingSequence+
    reducedRecHitsSequence+
    famosBTaggingSequence+
    famosPFTauTaggingSequence
)

# Simulation plus reconstruction
famosWithEverything = cms.Sequence(
    simulationWithFamos+
    reconstructionWithFamos
)

