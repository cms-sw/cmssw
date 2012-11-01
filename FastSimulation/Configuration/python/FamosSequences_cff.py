import FWCore.ParameterSet.Config as cms

from FastSimulation.Configuration.CommonInputs_cff import *

# Conversion to GenParticleCandidates 
from PhysicsTools.HepMCCandAlgos.genParticleCandidatesFast_cfi import *

# Pile-up options: FAMOS-style or with the Mixing Module
#the sequence is defined as a p[lace holder, to be defined separatedly
#from FastSimulation.Configuration.MixingFamos_cff import *
#from FastSimulation.Configuration.MixingFull_cff import *

# Famos SimHits producer
from FastSimulation.EventProducer.FamosSimHits_cff import *

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
# this one is added before 340pre3 to cope with adding SiPixelTemplateDBObjectESProducer and corresponding objects to the ConfDB (MC_3XY_V11, STARTUP3X_V10)
from CalibTracker.SiPixelESProducers.SiPixelTemplateDBObjectESProducer_cfi import *
from FastSimulation.Tracking.GlobalPixelTracking_cff import *
from FastSimulation.Tracking.IterativeTracking_cff import *

# Calo RecHits producer (with no HCAL miscalibration by default)
from FastSimulation.CaloRecHitsProducer.CaloRecHits_cff import *
from RecoLocalCalo.HcalRecAlgos.hcalRecAlgoESProd_cfi import *

# ECAL clusters
from RecoEcal.Configuration.RecoEcal_cff import *
reducedEcalRecHitsSequence.remove(seldigis)
# HCAL clusters
from RecoLocalCalo.HcalRecProducers.HcalHitSelection_cfi import *
reducedHcalRecHitsSequence = cms.Sequence( reducedHcalRecHits )

reducedRecHits = cms.Sequence ( reducedEcalRecHitsSequence * reducedHcalRecHitsSequence )


# Calo Towers
from RecoJets.Configuration.CaloTowersRec_cff import *

# Particle Flow (all interactions with ParticleFlow are dealt with in the following configuration)
#from FastSimulation.ParticleFlow.ParticleFlowFastSim_cff import *
from FastSimulation.ParticleFlow.ParticleFlowFastSimNeutralHadron_cff import * # this is the famous "PF patch", see https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideFastSimFAQ#I_observe_a_discrepancy_in_energ


# Reco Jets and MET
from RecoJets.Configuration.RecoJetsGlobal_cff import *
from RecoMET.Configuration.RecoMET_cff import *
metreco.remove(BeamHaloId)

caloJetMet = cms.Sequence(
    recoJets+
    recoJetIds+
    recoTrackJets+
    recoJetAssociations+recoJPTJets+
    metreco
)



# Gen Jets
from PhysicsTools.HepMCCandAlgos.genParticles_cfi import *
from RecoJets.Configuration.GenJetParticles_cff import *
from RecoJets.Configuration.RecoGenJets_cff import *
from RecoMET.Configuration.GenMETParticles_cff import *
from RecoMET.Configuration.RecoGenMET_cff import *
caloJetMetGen = cms.Sequence(
    genParticles+
    genJetParticles+
    recoGenJets+
    genMETParticles+
    recoGenMET
)


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
dt1DCosmicRecHits.dtDigiLabel = 'simMuonDTDigis'

# Muon reconstruction sequence
from RecoMuon.TrackingTools.MuonServiceProxy_cff import *
from RecoMuon.TrackingTools.MuonTrackLoader_cff import *
KFSmootherForMuonTrackLoader.Propagator = 'SmartPropagatorAny'
from RecoMuon.MuonSeedGenerator.standAloneMuonSeeds_cff import *
from RecoMuon.StandAloneMuonProducer.standAloneMuons_cff import *
from RecoMuon.Configuration.RecoMuonPPonly_cff import refittedStandAloneMuons
from FastSimulation.Configuration.globalMuons_cff import *
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
    refittedStandAloneMuons+    
    globalMuons+
    tevMuons
)

#Muon identification sequence
from RecoMuon.MuonIdentification.muonIdProducerSequence_cff import *

# Muon isolation
from RecoMuon.MuonIsolationProducers.muIsolation_cff import *

famosMuonIdAndIsolationSequence = cms.Sequence(
    ak5CaloJets+
    muonIdProducerSequence+
    muIsolation
)

from RecoMuon.MuonIdentification.muons_cfi import *
muons.FillSelectorMaps = False
muons.FillCosmicsIdMap = False
from RecoMuon.MuonIsolation.muonPFIsolation_cff import *

muonshighlevelreco = cms.Sequence(muonPFIsolationSequence*muons)


# Electron reconstruction
from FastSimulation.Tracking.globalCombinedSeeds_cfi import *
from RecoEgamma.EgammaHFProducers.hfEMClusteringSequence_cff import *
from RecoEgamma.EgammaElectronProducers.ecalDrivenElectronSeeds_cfi import *
from FastSimulation.EgammaElectronAlgos.electronGSGsfTrackCandidates_cff import *
from RecoEgamma.EgammaElectronProducers.gsfElectronSequence_cff import *
from TrackingTools.GsfTracking.GsfElectronFit_cff import *
from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
from RecoEgamma.EgammaPhotonProducers.allConversionSequence_cff import *
from RecoEgamma.Configuration.RecoEgamma_cff import egammaHighLevelRecoPostPF
allConversions.src = 'gsfGeneralConversionTrackMerger'
famosConversionSequence = cms.Sequence(conversionTrackSequenceNoEcalSeeded*allConversionSequence)

from TrackingTools.GsfTracking.CkfElectronCandidateMaker_cff import *
from TrackingTools.GsfTracking.FwdElectronPropagator_cfi import *
import TrackingTools.GsfTracking.GsfElectronFit_cfi

egammaEcalDrivenReco = cms.Sequence(gsfEcalDrivenElectronSequence)
electronGsfTracks = TrackingTools.GsfTracking.GsfElectronFit_cfi.GsfGlobalElectronTest.clone()
electronGsfTracks.src = 'electronGSGsfTrackCandidates'
electronGsfTracks.TTRHBuilder = 'WithoutRefit'
electronGsfTracks.TrajectoryInEvent = True


# PF related electron sequences defined in FastSimulation.ParticleFlow.ParticleFlowFastSim_cff
from RecoEgamma.ElectronIdentification.electronIdSequence_cff import *

iterativeTrackingBeginning = cms.Sequence(
    iterativeInitialSeeds+
    iterativePixelPairSeeds+
    iterativeMixedTripletStepSeeds+
    iterativePixelLessSeeds
    )

famosGsfTrackSequence = cms.Sequence(
    iterativeTrackingBeginning+ 
    newCombinedSeeds+
    particleFlowCluster+ 
    ecalDrivenElectronSeeds+
    trackerDrivenElectronSeeds+
    electronMergedSeeds+
    electronGSGsfTrackCandidates+
    electronGsfTracks
)

# Photon reconstruction
from RecoEgamma.EgammaPhotonProducers.photonSequence_cff import *
photons.hbheInstance = ''
from RecoEgamma.PhotonIdentification.photonId_cff import *

famosPhotonSequence = cms.Sequence(
    photonSequence+
    photonIDSequence
)

# Add pre-calculated isolation sums for electrons (NB for photons they are stored in the Photon. All is done in the
# sequence above
from RecoEgamma.EgammaIsolationAlgos.egammaIsolationSequence_cff import *

#Add egamma ecal interesting rec hits
from RecoEgamma.EgammaIsolationAlgos.interestingEleIsoDetIdModule_cff import *
from RecoEgamma.EgammaIsolationAlgos.interestingGamIsoDetIdModule_cff import *

from RecoEgamma.EgammaIsolationAlgos.interestingEgammaIsoDetIdsSequence_cff import *



# B tagging
from RecoJets.JetAssociationProducers.ak5JTA_cff import *
ak5JetTracksAssociatorAtVertex.tracks = 'generalTracks'
from RecoVertex.Configuration.RecoVertex_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoBTag.Configuration.RecoBTag_cff import *

famosBTaggingSequence = cms.Sequence(
    btagging
)



# The sole simulation sequence
simulationSequence = cms.Sequence(
    offlineBeamSpot+
    cms.SequencePlaceholder("famosMixing")+
    famosSimHits+
    MuonSimHits+
    cms.SequencePlaceholder("mix")
    )

# Calo simulation mode is defined in FastSimulation/CaloRecHitsProducer/python/CaloRecHits_cff.py
if(CaloMode==0):
    lowLevelRecoSequence = cms.Sequence(
        caloRecHits
        )
elif(CaloMode==1):
    lowLevelRecoSequence = cms.Sequence(
        caloDigis+
        caloRecHits 
        )
else:
    lowLevelRecoSequence = cms.Sequence(
        siTrackerGaussianSmearingRecHits+
        iterativeTracking+ # because tracks are used for noise cleaning in HCAL low-level reco
        caloDigis+
        caloRecHits 
        )

famosSimulationSequence = cms.Sequence(
    simulationSequence+
    lowLevelRecoSequence
    )

famosEcalDrivenElectronSequence = cms.Sequence(
    famosGsfTrackSequence+
    egammaEcalDrivenReco
)

# The reconstruction sequence
reconstructionWithFamos = cms.Sequence(
    cms.SequencePlaceholder("mix")+
    iterativeTracking+
    vertexreco+
    caloTowersRec+
    ecalClusters+
    particleFlowCluster+
    famosGsfTrackSequence+
    famosMuonSequence+
    famosMuonIdAndIsolationSequence+
    famosConversionSequence+
    particleFlowTrackWithDisplacedVertex+
    famosEcalDrivenElectronSequence+
    famosPhotonSequence+
    famosParticleFlowSequence+
    egammaHighLevelRecoPostPF+
    muonshighlevelreco+
    particleFlowLinks+
    caloJetMetGen+
    caloJetMet+
    PFJetMet+
    ic5JetTracksAssociatorAtVertex+
    ak5JetTracksAssociatorAtVertex+
    famosTauTaggingSequence+
    reducedRecHits+
    famosBTaggingSequence+
    famosPFTauTaggingSequence
)

# Famos pre-defined sequences (and self-explanatory names)

famosWithTrackerHits = cms.Sequence(
    famosSimulationSequence+
    siTrackerGaussianSmearingRecHits
)

famosWithTracks = cms.Sequence(
    famosWithTrackerHits+
    iterativeTracking
)

famosWithTracksAndMuonHits = cms.Sequence(
    famosSimulationSequence+
    siTrackerGaussianSmearingRecHits+
    iterativeTracking+
    vertexreco+
    famosMuonSequence
)

famosWithTracksAndMuons = cms.Sequence(
    famosSimulationSequence+
    siTrackerGaussianSmearingRecHits+
    iterativeTracking+
    vertexreco+
    famosMuonSequence+
    caloTowersRec+
    famosMuonIdAndIsolationSequence
)

famosWithCaloHits = cms.Sequence(
    famosSimulationSequence+
    caloTowersRec
)

famosWithEcalClusters = cms.Sequence(
    famosWithCaloHits+
    ecalClusters+
    particleFlowCluster
)

famosWithTracksAndCaloHits = cms.Sequence(
    famosWithTracks+
    caloTowersRec
)

famosWithTracksAndEcalClusters = cms.Sequence(
    famosWithTracksAndCaloHits+
    ecalClusters+
    particleFlowCluster
)

    
famosWithParticleFlow = cms.Sequence(
    famosWithTracksAndEcalClusters+
    vertexreco+
    famosGsfTrackSequence+
    famosConversionSequence+
    caloTowersRec+ 
    famosParticleFlowSequence+
    PFJetMet
)

famosWithCaloTowers = cms.Sequence(
    famosWithCaloHits+
    caloTowersRec
)

famosElectronSequence = cms.Sequence(
    famosGsfTrackSequence+
    famosEcalDrivenElectronSequence+
    famosWithParticleFlow+
    egammaHighLevelRecoPostPF+
    gsfElectronSequence+
    eIdSequence
)

famosWithTracksAndCaloTowers = cms.Sequence(
    famosWithTracksAndCaloHits+
    caloTowersRec
)

famosWithTracksAndJets = cms.Sequence(
    famosWithTracksAndCaloTowers+
    vertexreco+
    caloJetMetGen+
    caloJetMet
)

### Standard Jets _cannot_ be done without many other things...
#######################################################################
famosWithJets = cms.Sequence(
    famosWithTracksAndCaloTowers+
    vertexreco+
    ecalClusters+
    particleFlowCluster+
    famosGsfTrackSequence+
    famosMuonSequence+
    famosMuonIdAndIsolationSequence+
    famosParticleFlowSequence+
    gsfElectronSequence+	
    caloJetMetGen+
    caloJetMet
)

##--- simplified IC05 jets only
#famosWithSimpleJets = cms.Sequence(
#    famosWithTracksAndCaloTowers+
#    vertexreco+
#    caloJetMetGen+
#    iterativeCone5CaloJets+
#    ic5JetTracksAssociatorAtVertex
#)

famosWithCaloTowersAndParticleFlow = cms.Sequence(
    famosWithParticleFlow+
#    caloTowersRec
    caloTowersRec
)


famosWithElectrons = cms.Sequence(
    famosWithTracksAndEcalClusters+
#    caloTowersRec+
    caloTowersRec+
    famosGsfTrackSequence+
    famosParticleFlowSequence+
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
#    caloRecHits+
    caloTowersRec+
    ecalClusters+
#    caloTowersRec+
    famosElectronSequence+
    famosPhotonSequence+
    interestingEgammaIsoDetIds+
    egammaIsolationSequence
)

famosWithBTagging = cms.Sequence(
    famosWithTracksAndCaloTowers+
    vertexreco+
    ak5CaloJets+
    ak5JetTracksAssociatorAtVertex+
    ecalClusters+
    famosMuonSequence+
    reducedRecHits+ 
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

# The simulation sequence without muon digitization
simulationNoMuonDigiWithFamos = cms.Sequence(
    famosSimulationSequence+
    siTrackerGaussianSmearingRecHits
)

# The simulation and digitization sequence
simulationWithFamos = cms.Sequence(
    famosSimulationSequence+
    muonDigi+
    siTrackerGaussianSmearingRecHits
)



reconstructionWithFamosNoTk = cms.Sequence(
    vertexreco+
    caloRecHits+
    caloTowersRec+
    ecalClusters+
    particleFlowCluster+
    famosGsfTrackSequence+
    famosMuonSequence+
    famosMuonIdAndIsolationSequence+
    famosConversionSequence+
    particleFlowTrackWithDisplacedVertex+
    famosEcalDrivenElectronSequence+
    famosPhotonSequence+
    famosParticleFlowSequence+
    egammaHighLevelRecoPostPF+
    muonshighlevelreco+
    caloJetMetGen+
    caloJetMet+
    PFJetMet+
    ic5JetTracksAssociatorAtVertex+
    ak5JetTracksAssociatorAtVertex+
    famosTauTaggingSequence+
    reducedRecHits+
    famosBTaggingSequence+
    famosPFTauTaggingSequence
)

# Simulation plus reconstruction
famosWithEverything = cms.Sequence(
    simulationWithFamos+
    reconstructionWithFamos
)

