import FWCore.ParameterSet.Config as cms

#Particle data table, Magnetic Field, CMS geometry, Tracker geometry, Calo geometry
#include "FastSimulation/Configuration/data/CommonInputs.cff"
# Primary vertex smearing.
from IOMC.EventVertexGenerators.VtxSmearedGauss_cfi import *
# Conversion to GenParticleCandidates 
from PhysicsTools.HepMCCandAlgos.genParticleCandidatesFast_cfi import *
# Famos PileUp Producer
from FastSimulation.PileUpProducer.PileUpProducer_cff import *
# Famos SimHits producer
from FastSimulation.EventProducer.FamosSimHits_cff import *
# Mixing module 
from SimGeneral.MixingModule.mixNoPU_cfi import *
#include "SimGeneral/MixingModule/data/mixLowLumPU.cfi"
#include "SimGeneral/MixingModule/data/mixHighPU.cfi"
# Gaussian Smearing RecHit producer
from FastSimulation.TrackingRecHitProducer.SiTrackerGaussianSmearingRecHitConverter_cfi import *
# Rec Hit Tranlator to the Full map with DeTId'
from FastSimulation.TrackingRecHitProducer.TrackingRecHitTranslator_cfi import *
# CTF and Iterative tracking (contains pixelTracks and pixelVertices)
# 1) Common algorithms and configuration taken from full reconstruction
# Note: The runge-kutta propagator is not used here 
# (because no magnetic field inhomogeneities are simulated between layers)
from FastSimulation.Tracking.GSTrackFinalFitCommon_cff import *
# 3) Fast Simulation tracking sequences
from FastSimulation.Tracking.GlobalPixelTracking_cff import *
from FastSimulation.Tracking.IterativeTracking_cff import *
# Calo RecHits producer (with no HCAL miscalibration by default)
from FastSimulation.CaloRecHitsProducer.CaloRecHits_cff import *
# ECAL clusters (this will generate harmless warnings...)
from RecoEcal.EgammaClusterProducers.ecalClusteringSequence_cff import *
# Particle Flow
from RecoParticleFlow.PFClusterProducer.towerMakerPF_cff import *
from RecoParticleFlow.PFClusterProducer.particleFlowCluster_cff import *
from RecoParticleFlow.PFTracking.particleFlowTrack_cff import *
from RecoParticleFlow.PFBlockProducer.particleFlowSimParticle_cff import *
from RecoParticleFlow.PFBlockProducer.particleFlowBlock_cff import *
from RecoParticleFlow.PFProducer.particleFlow_cff import *
import copy
from FastSimulation.Tracking.TrackCandidateProducer_cfi import *
fsGsfElCandidates = copy.deepcopy(trackCandidateProducer)
import copy
from TrackingTools.GsfTracking.GsfElectronFit_cfi import *
fsgsfPFtracks = copy.deepcopy(GsfGlobalElectronTest)
from Configuration.JetMET.calorimetry_jetmet_gen_cff import *
from Configuration.JetMET.calorimetry_jetmet_cff import *
# Muon simHit sequence 
from FastSimulation.MuonSimHitProducer.MuonSimHitProducer_cfi import *
# Muon Digi sequence
from SimMuon.Configuration.SimMuon_cff import *
# Muon RecHit sequence
from RecoLocalMuon.Configuration.RecoLocalMuon_cff import *
#replace  DTParametrizedDriftAlgo_CSA07.recAlgoConfig.tTrigModeConfig.doT0Correction = false
# Muon reconstruction sequence
#include "RecoMuon/Configuration/data/RecoMuon.cff"
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *
from RecoMuon.MuonSeedGenerator.standAloneMuonSeeds_cfi import *
from RecoMuon.StandAloneMuonProducer.standAloneMuons_cfi import *
from FastSimulation.Configuration.globalMuons_cff import *
#Muon identification sequence
#include "RecoMuon/MuonIdentification/data/muonIdProducerSequence.cff"
from FastSimulation.Configuration.muonIdentification_cff import *
#Muon isolation
from RecoMuon.MuonIsolationProducers.muIsolation_cff import *
# Electron reconstruction
from FastSimulation.EgammaElectronAlgos.electronGSPixelSeeds_cfi import *
from FastSimulation.EgammaElectronAlgos.electronGSGsfTrackCandidates_cff import *
from RecoEgamma.EgammaElectronProducers.pixelMatchGsfElectrons_cff import *
from TrackingTools.GsfTracking.GsfElectronFit_cff import *
import copy
from TrackingTools.GsfTracking.GsfElectronFit_cfi import *
pixelMatchGsfFit = copy.deepcopy(GsfGlobalElectronTest)
# Photon reconstruction
from RecoEgamma.EgammaPhotonProducers.photonSequence_cff import *
# B tagging
from RecoJets.JetAssociationProducers.ic5JetTracksAssociatorAtVertex_cfi import *
from RecoVertex.Configuration.RecoVertex_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoBTag.Configuration.RecoBTag_cff import *
#Tau tagging
from RecoTauTag.Configuration.RecoTauTag_cff import *
from RecoTauTag.Configuration.RecoPFTauTag_cff import *
famosParticleFlowSequence = cms.Sequence(caloTowersPFRec+particleFlowCluster+elecpreid+fsGsfElCandidates+fsgsfPFtracks+pfTrackElec+particleFlowBlock+particleFlow)
famosMuonSequence = cms.Sequence(muonDigi+muonlocalreco+MuonSeed+standAloneMuons+globalMuons)
#replace muIsoDepositTk.ExtractorPSet = { using MIsoTrackExtractorGsBlock }
famosMuonIdAndIsolationSequence = cms.Sequence(muonIdProducerSequence+sisCone5CaloJets+muIsolation)
famosElectronSequence = cms.Sequence(electronGSPixelSeeds+electronGSGsfTrackCandidates+pixelMatchGsfFit+pixelMatchGsfElectrons)
famosPhotonSequence = cms.Sequence(photonSequence)
famosBTaggingSequence = cms.Sequence(impactParameterTagInfos*jetBProbabilityBJetTags+jetProbabilityBJetTags+trackCountingHighPurBJetTags+trackCountingHighEffBJetTags+impactParameterMVABJetTags*secondaryVertexTagInfos*simpleSecondaryVertexBJetTags+combinedSecondaryVertexBJetTags+combinedSecondaryVertexMVABJetTags)
famosTauTaggingSequence = cms.Sequence(tautagging)
famosPFTauTaggingSequence = cms.Sequence(PFTau)
# The sole simulation sequence
famosSimulationSequence = cms.Sequence(offlineBeamSpot+famosPileUp+famosSimHits+genParticleCandidates+MuonSimHits+mix)
# Famos pre-defined sequences
famosWithTrackerHits = cms.Sequence(famosSimulationSequence+siTrackerGaussianSmearingRecHits)
famosWithTrackerAndCaloHits = cms.Sequence(famosWithTrackerHits+caloRecHits)
famosWithTracks = cms.Sequence(famosWithTrackerHits+iterativeTracking)
famosWithTracksAndMuonHits = cms.Sequence(famosSimulationSequence+siTrackerGaussianSmearingRecHits+iterativeTracking+famosMuonSequence)
famosWithTracksAndMuons = cms.Sequence(famosSimulationSequence+siTrackerGaussianSmearingRecHits+iterativeTracking+famosMuonSequence+caloRecHits+cms.SequencePlaceholder("towerMaker")+cms.SequencePlaceholder("caloTowers")+famosMuonIdAndIsolationSequence)
famosWithCaloHits = cms.Sequence(famosSimulationSequence+caloRecHits)
famosWithEcalClusters = cms.Sequence(famosWithCaloHits+ecalClusteringSequence)
famosWithTracksAndCaloHits = cms.Sequence(famosWithTracks+caloRecHits)
famosWithTracksAndEcalClusters = cms.Sequence(famosWithTracksAndCaloHits+ecalClusteringSequence)
famosWithParticleFlow = cms.Sequence(famosWithTracksAndCaloHits+famosParticleFlowSequence)
famosWithCaloTowers = cms.Sequence(famosWithCaloHits+cms.SequencePlaceholder("towerMaker")+cms.SequencePlaceholder("caloTowers"))
famosWithJets = cms.Sequence(famosWithCaloTowers+caloJetMetGen+caloJetMet)
famosWithTracksAndCaloTowers = cms.Sequence(famosWithTracksAndCaloHits+cms.SequencePlaceholder("towerMaker")+cms.SequencePlaceholder("caloTowers"))
famosWithTracksAndJets = cms.Sequence(famosWithTracksAndCaloTowers+caloJetMetGen+caloJetMet)
famosWithCaloTowersAndParticleFlow = cms.Sequence(famosWithParticleFlow+cms.SequencePlaceholder("towerMaker")+cms.SequencePlaceholder("caloTowers"))
famosWithMuons = cms.Sequence(famosWithTracks+cms.SequencePlaceholder("paramMuons"))
famosWithMuonsAndIsolation = cms.Sequence(famosWithTracksAndCaloTowers+cms.SequencePlaceholder("paramMuons")+sisCone5CaloJets+muIsolation_ParamGlobalMuons)
famosWithElectrons = cms.Sequence(famosWithTrackerHits+caloRecHits+ecalClusteringSequence+famosElectronSequence)
famosWithPhotons = cms.Sequence(famosWithTrackerHits+caloRecHits+ecalClusteringSequence+famosPhotonSequence)
famosWithElectronsAndPhotons = cms.Sequence(famosWithTrackerHits+caloRecHits+ecalClusteringSequence+famosElectronSequence+famosPhotonSequence)
famosWithBTagging = cms.Sequence(famosWithTracksAndCaloTowers+vertexreco+iterativeCone5CaloJets+ic5JetTracksAssociatorAtVertex+famosBTaggingSequence)
famosWithTauTagging = cms.Sequence(famosWithTracksAndCaloTowers+vertexreco+iterativeCone5CaloJets+ic5JetTracksAssociatorAtVertex+ecalClusteringSequence+famosTauTaggingSequence)
famosWithPFTauTagging = cms.Sequence(famosWithCaloTowersAndParticleFlow+vertexreco+famosPFTauTaggingSequence)
famosWithEverything = cms.Sequence(famosWithCaloTowersAndParticleFlow+vertexreco+ecalClusteringSequence+famosElectronSequence+famosPhotonSequence+famosMuonSequence+famosMuonIdAndIsolationSequence+caloJetMetGen+caloJetMet+cms.SequencePlaceholder("paramMuons")+muIsolation_ParamGlobalMuons+ic5JetTracksAssociatorAtVertex+famosBTaggingSequence+famosTauTaggingSequence+famosPFTauTaggingSequence)
# The simulation sequence
simulationWithFamos = cms.Sequence(famosSimulationSequence+siTrackerGaussianSmearingRecHits+caloRecHits)
# The reconstruction sequence
reconstructionWithFamos = cms.Sequence(iterativeTracking+vertexreco+cms.SequencePlaceholder("towerMaker")+cms.SequencePlaceholder("caloTowers")+ecalClusteringSequence+famosElectronSequence+famosPhotonSequence+famosMuonSequence+famosMuonIdAndIsolationSequence+famosParticleFlowSequence+caloJetMetGen+caloJetMet+cms.SequencePlaceholder("paramMuons")+muIsolation_ParamGlobalMuons+ic5JetTracksAssociatorAtVertex+famosBTaggingSequence+famosTauTaggingSequence+famosPFTauTaggingSequence)
# 2) Specific cuts
# Add a chi**2 cut to retain/reject hits
KFFittingSmoother.EstimateCut = 15.0
# Request three hits to make a track
KFFittingSmoother.MinNumberOfHits = 3
islandBasicClusters.barrelHitProducer = 'caloRecHits'
islandBasicClusters.endcapHitProducer = 'caloRecHits'
hybridSuperClusters.ecalhitproducer = 'caloRecHits'
correctedHybridSuperClusters.recHitProducer = 'caloRecHits'
correctedIslandBarrelSuperClusters.recHitProducer = 'caloRecHits'
correctedIslandEndcapSuperClusters.recHitProducer = 'caloRecHits'
correctedEndcapSuperClustersWithPreshower.preshRecHitProducer = 'caloRecHits'
preshowerClusterShape.preshRecHitProducer = 'caloRecHits'
dynamicHybridSuperClusters.ecalhitproducer = 'caloRecHits'
correctedDynamicHybridSuperClusters.recHitProducer = 'caloRecHits'
fixedMatrixBasicClusters.barrelHitProducer = 'caloRecHits'
fixedMatrixBasicClusters.endcapHitProducer = 'caloRecHits'
fixedMatrixPreshowerClusterShape.preshRecHitProducer = 'caloRecHits'
fixedMatrixSuperClustersWithPreshower.preshRecHitProducer = 'caloRecHits'
correctedFixedMatrixSuperClustersWithPreshower.recHitProducer = 'caloRecHits'
towerMakerPF.ecalInputs = cms.VInputTag(cms.InputTag("caloRecHits","EcalRecHitsEB"), cms.InputTag("caloRecHits","EcalRecHitsEE"))
towerMakerPF.hbheInput = 'caloRecHits'
towerMakerPF.hoInput = 'caloRecHits'
towerMakerPF.hfInput = 'caloRecHits'
particleFlowRecHitECAL.ecalRecHitsEB = cms.InputTag("caloRecHits","EcalRecHitsEB")
particleFlowRecHitECAL.ecalRecHitsEE = cms.InputTag("caloRecHits","EcalRecHitsEE")
particleFlowRecHitPS.ecalRecHitsES = cms.InputTag("caloRecHits","EcalRecHitsES")
particleFlowSimParticle.sim = 'famosSimHits'
elecpreid.NHitsInSeed = 1
fsGsfElCandidates.SeedProducer = cms.InputTag("elecpreid","SeedsForGsf")
fsGsfElCandidates.TrackProducer = cms.InputTag("None","None")
fsGsfElCandidates.MinNumberOfCrossedLayers = 5
fsgsfPFtracks.src = 'fsGsfElCandidates'
fsgsfPFtracks.TTRHBuilder = 'WithoutRefit'
fsgsfPFtracks.TrajectoryInEvent = True
pfTrackElec.GsfTrackModuleLabel = 'fsgsfPFtracks'
genParticleCandidates.src = 'source'
genCandidatesForMET.verbose = False
muonCSCDigis.strips.doCorrelatedNoise = False
GlobalTrajectoryBuilderCommon.TrackRecHitBuilder = 'WithoutRefit'
GlobalTrajectoryBuilderCommon.TrackTransformer.TrackerRecHitBuilder = 'WithoutRefit'
globalMuons.TrackerCollectionLabel = 'generalTracks'
# Use FastSim tracks and calo hits for muon id
muons.inputCollectionLabels = cms.VInputTag('generalTracks', 'globalMuons', cms.InputTag("standAloneMuons","UpdatedAtVtx"))
muons.TrackAssociatorParameters.EBRecHitCollectionLabel = cms.InputTag("caloRecHits","EcalRecHitsEB")
muons.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("caloRecHits","EcalRecHitsEE")
muons.TrackAssociatorParameters.CaloTowerCollectionLabel = 'towerMaker'
muons.TrackAssociatorParameters.HBHERecHitCollectionLabel = 'caloRecHits'
muons.TrackAssociatorParameters.HORecHitCollectionLabel = 'caloRecHits'
#replace muons.TrackExtractorPSet = { using MIsoTrackExtractorGsBlock }
# Use FastSim tracks and calo hits for calo muon id
calomuons.inputTracks = 'generalTracks'
calomuons.TrackAssociatorParameters.EBRecHitCollectionLabel = cms.InputTag("caloRecHits","EcalRecHitsEB")
calomuons.TrackAssociatorParameters.EERecHitCollectionLabel = cms.InputTag("caloRecHits","EcalRecHitsEE")
calomuons.TrackAssociatorParameters.CaloTowerCollectionLabel = 'towerMaker'
calomuons.TrackAssociatorParameters.HBHERecHitCollectionLabel = 'caloRecHits'
calomuons.TrackAssociatorParameters.HORecHitCollectionLabel = 'caloRecHits'
pixelMatchGsfFit.src = 'electronGSGsfTrackCandidates'
pixelMatchGsfFit.TTRHBuilder = 'WithoutRefit'
pixelMatchGsfElectrons.hcalRecHits = 'caloRecHits'
pixelMatchGsfElectrons.barrelSuperClusters = cms.InputTag("correctedHybridSuperClusters","electronGSPixelSeeds")
pixelMatchGsfElectrons.endcapSuperClusters = cms.InputTag("correctedEndcapSuperClustersWithPreshower","electronGSPixelSeeds")
photons.barrelHitProducer = 'caloRecHits'
photons.endcapHitProducer = 'caloRecHits'
photons.hbheModule = 'caloRecHits'
photons.hbheInstance = ''
photons.pixelSeedProducer = 'electronGSPixelSeeds'
ic5JetTracksAssociatorAtVertex.tracks = 'generalTracks'
ic5PFJetTracksAssociatorAtVertex.tracks = 'generalTracks'
offlinePrimaryVerticesFromCTFTracks.TrackLabel = 'generalTracks'
#replace combinedTauTag.PVSrc = "offlinePrimaryVerticesFromCTFTracks"
caloRecoTauProducer.PVProducer = 'offlinePrimaryVerticesFromCTFTracks'
caloRecoTauTagInfoProducer.EBRecHitsSource = cms.InputTag("caloRecHits","EcalRecHitsEB")
caloRecoTauTagInfoProducer.EERecHitsSource = cms.InputTag("caloRecHits","EcalRecHitsEE")
pfRecoTauProducer.PVProducer = 'offlinePrimaryVerticesFromCTFTracks'

