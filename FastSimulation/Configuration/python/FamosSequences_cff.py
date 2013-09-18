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

caloJets = cms.Sequence(
    recoJets+
    recoJetIds+
    recoTrackJets
)

jetTrackAssoc = cms.Sequence (
    recoJetAssociations
    )

jetPlusTracks = cms.Sequence(
    recoJPTJets
    )

metReco = cms.Sequence(
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


if (MixingMode==2):
    simMuonCSCDigis.mixLabel = 'mixSimCaloHits'
    simMuonDTDigis.mixLabel = 'mixSimCaloHits'
    simMuonRPCDigis.mixLabel = 'mixSimCaloHits'
else:
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

if (CaloMode==3):
    famosMuonSequence = cms.Sequence(
        muonlocalreco+
        ancientMuonSeed+
        standAloneMuons+
        refittedStandAloneMuons+    
        globalMuons+
        tevMuons
        )
else:
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
from RecoEgamma.EgammaElectronProducers.gedGsfElectronSequence_cff import *
from TrackingTools.GsfTracking.GsfElectronFit_cff import *
from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
from RecoEgamma.EgammaPhotonProducers.allConversionSequence_cff import *
from RecoEgamma.EgammaPhotonProducers.gedPhotonSequence_cff import *
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
from RecoVertex.Configuration.RecoVertex_cff import *
from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoBTag.Configuration.RecoBTag_cff import *

famosBTaggingSequence = cms.Sequence(
    btagging
)


############### FastSim sequences

# Calo simulation mode is defined in FastSimulation/CaloRecHitsProducer/python/CaloRecHits_cff.py
if(CaloMode==0):
    simulationSequence = cms.Sequence(
        offlineBeamSpot+
        cms.SequencePlaceholder("famosMixing")+
        famosSimHits+
        MuonSimHits+
        cms.SequencePlaceholder("mix")
        )
    lowLevelRecoSequence = cms.Sequence(
        siTrackerGaussianSmearingRecHits+
        caloRecHits
        )
    famosSimulationSequence = cms.Sequence(
        simulationSequence+
        lowLevelRecoSequence
        )
    trackVertexReco = cms.Sequence(
        cms.SequencePlaceholder("mix")+
        iterativeTracking+
        vertexreco
        )
    caloTowersSequence = cms.Sequence(
        caloTowersRec
        )
elif(CaloMode==1):
    simulationSequence = cms.Sequence(
        offlineBeamSpot+
        cms.SequencePlaceholder("famosMixing")+
        famosSimHits+
        MuonSimHits+
        cms.SequencePlaceholder("mix")
        )
    lowLevelRecoSequence = cms.Sequence(
        siTrackerGaussianSmearingRecHits+
        caloDigis+
        caloRecHits 
        )
    famosSimulationSequence = cms.Sequence(
        simulationSequence+
        lowLevelRecoSequence
        )
    trackVertexReco = cms.Sequence(
        cms.SequencePlaceholder("mix")+
        iterativeTracking+
        vertexreco
        )
    caloTowersSequence = cms.Sequence(
        caloTowersRec
        )
elif(CaloMode==2):
    simulationSequence = cms.Sequence(
        offlineBeamSpot+
        cms.SequencePlaceholder("famosMixing")+
        famosSimHits+
        MuonSimHits+
        cms.SequencePlaceholder("mix")
        )
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
    trackVertexReco = cms.Sequence(
        cms.SequencePlaceholder("mix")+
        iterativeTracking+ # this repetition is normally harmless, but it is annoying if you want to run SIM and RECO in two steps
        vertexreco
        )
    caloTowersSequence = cms.Sequence(
        caloTowersRec
        )
elif(CaloMode==3):
    if(MixingMode==1):
        simulationSequence = cms.Sequence(
            offlineBeamSpot+
            cms.SequencePlaceholder("famosMixing")+
            famosSimHits+
            MuonSimHits
            )
        digitizationSequence = cms.Sequence(
            cms.SequencePlaceholder("mix")+
            muonDigi+
            caloDigis
            )
        trackVertexReco = cms.Sequence(
            siTrackerGaussianSmearingRecHits+
            iterativeTracking+ 
            vertexreco
            )
    else:
        simulationSequence = cms.Sequence(
            offlineBeamSpot+
            famosSimHits+
            MuonSimHits
            )
        digitizationSequence = cms.Sequence(
            cms.SequencePlaceholder("mixHits")+
            muonDigi+
            caloDigis
            )
        trackVertexReco = cms.Sequence(
            siTrackerGaussianSmearingRecHits+
            iterativeTracking+ 
            cms.SequencePlaceholder("mixTracks")+ 
            vertexreco
            )
# out of the 'if':
    caloTowersSequence = cms.Sequence(
        caloRecHits+
        caloTowersRec
        )
    famosSimulationSequence = cms.Sequence( # valid for both MixingMode values
        simulationSequence+
        digitizationSequence#+ # temporary; eventually it will be a block of its own, but it requires intervention on ConfigBuilder
        # Note: of course it is a bit odd that the next two sequences are made part of the SIM step, but this is a temporary solution needed because currently HLT is run before reconstructionWithFamos, and HLT needs to access the caloRecHits, which in turn depend on tracks because HCAL hits use the TrackExtrapolator
#        trackVertexReco+
#        caloTowersSequence
        )


famosEcalDrivenElectronSequence = cms.Sequence(
    famosGsfTrackSequence+
    egammaEcalDrivenReco
)

# The reconstruction sequence
if(CaloMode==3):
    reconstructionWithFamos = cms.Sequence(
        digitizationSequence+ # temporary; repetition!
        trackVertexReco+
        caloTowersSequence+
        particleFlowCluster+
        ecalClusters+
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
        caloJets+
        PFJetMet+
        jetTrackAssoc+
        recoJPTJets+
        metreco+
        reducedRecHits+
        famosBTaggingSequence+
        famosPFTauTaggingSequence
        )
else:
    reconstructionWithFamos = cms.Sequence(
        trackVertexReco+
        caloTowersSequence+
        particleFlowCluster+
        ecalClusters+
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
        caloJets+
        PFJetMet+
        jetTrackAssoc+
        recoJPTJets+
        metreco+
        reducedRecHits+
        famosBTaggingSequence+
        famosPFTauTaggingSequence
        )




# Special sequences for two-step operation
simulationWithSomeReconstruction = cms.Sequence(
    famosSimulationSequence+
    siTrackerGaussianSmearingRecHits+
    iterativeTracking+
    vertexreco+
    caloTowersSequence+
    particleFlowCluster+
    ecalClusters+
    famosGsfTrackSequence+
    famosMuonSequence+
    famosMuonIdAndIsolationSequence+
    famosConversionSequence+
    particleFlowTrackWithDisplacedVertex+
    famosEcalDrivenElectronSequence+
    famosPhotonSequence+
    famosParticleFlowSequence+
    egammaHighLevelRecoPostPF
    )

reconstructionHighLevel = cms.Sequence(
    cms.SequencePlaceholder("mix")+
    muonshighlevelreco+
    particleFlowLinks+
    caloJetMetGen+
    caloJets+
    PFJetMet+
    jetTrackAssoc+
    recoJPTJets+
    metreco+
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
    caloTowersSequence+
    famosMuonIdAndIsolationSequence
)

famosWithCaloHits = cms.Sequence(
    famosSimulationSequence+
    caloTowersSequence
)

famosWithEcalClusters = cms.Sequence(
    famosWithCaloHits+
    particleFlowCluster+
    ecalClusters
)

famosWithTracksAndCaloHits = cms.Sequence(
    famosWithTracks+
    caloTowersSequence
)

famosWithTracksAndEcalClusters = cms.Sequence(
    famosWithTracksAndCaloHits+
    particleFlowCluster+
    ecalClusters
)

    
famosWithParticleFlow = cms.Sequence(
    famosWithTracksAndEcalClusters+
    vertexreco+
    famosGsfTrackSequence+
    famosConversionSequence+
    caloTowersSequence+ 
    famosParticleFlowSequence+
    PFJetMet
)

famosWithCaloTowers = cms.Sequence(
    famosWithCaloHits+
    caloTowersSequence
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
    caloTowersSequence
)

famosWithTracksAndJets = cms.Sequence(
    famosWithTracksAndCaloTowers+
    vertexreco+
    caloJetMetGen+
    caloJets+
    metreco
)

### Standard Jets _cannot_ be done without many other things...
#######################################################################
famosWithJets = cms.Sequence(
    famosWithTracksAndCaloTowers+
    vertexreco+
    particleFlowCluster+
    ecalClusters+
    famosGsfTrackSequence+
    famosMuonSequence+
    famosMuonIdAndIsolationSequence+
    famosParticleFlowSequence+
    gsfElectronSequence+	
    caloJetMetGen+
    caloJets+
    metreco
)

famosWithCaloTowersAndParticleFlow = cms.Sequence(
    famosWithParticleFlow+
    caloTowersSequence
)


famosWithElectrons = cms.Sequence(
    famosWithTracksAndEcalClusters+
    caloTowersSequence+
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
    ecalClustersNoPFBox+
    famosPhotonSequence+
    interestingGamIsoDetIdEB+
    interestingGamIsoDetIdEE
)

famosWithElectronsAndPhotons = cms.Sequence(
    famosWithTracks+
    vertexreco+
    caloTowersSequence+
    ecalClustersNoPFBox+
    famosElectronSequence+
    famosPhotonSequence+
    interestingEgammaIsoDetIds+
    egammaIsolationSequence
)

famosWithBTagging = cms.Sequence(
    famosWithTracksAndCaloTowers+
    vertexreco+
    ak5PFJetsCHS+
    PFJetMet+
    jetTrackAssoc+
    metreco+
    ecalClustersNoPFBox+
    famosMuonSequence+
    reducedRecHits+ 
    famosBTaggingSequence
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
    famosSimulationSequence#+
#    muonDigi+
#    siTrackerGaussianSmearingRecHits
)



reconstructionWithFamosNoTk = cms.Sequence(
    vertexreco+
    caloTowersSequence+
    particleFlowCluster+
    ecalClusters+
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
    caloJets+
    metreco+
    PFJetMet+
    jetTrackAssoc+
    recoJPTJets+
    metreco+
    reducedRecHits+
    famosBTaggingSequence+
    famosPFTauTaggingSequence
)

# Simulation plus reconstruction
famosWithEverything = cms.Sequence(
    simulationWithFamos+
    reconstructionWithFamos
)

