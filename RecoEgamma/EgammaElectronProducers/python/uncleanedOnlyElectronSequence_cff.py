
#import copy
#from PhysicsTools.PatAlgos.tools.helpers import *

#
# Tracking
#

from RecoEgamma.EgammaElectronProducers.ecalDrivenElectronSeeds_cff import *
uncleanedOnlyElectronSeeds = ecalDrivenElectronSeeds.clone(
    barrelSuperClusters = "uncleanedOnlyCorrectedHybridSuperClusters",
    endcapSuperClusters = "uncleanedOnlyCorrectedMulti5x5SuperClustersWithPreshower"
    )

from TrackingTools.GsfTracking.CkfElectronCandidateMaker_cff import *
uncleanedOnlyElectronCkfTrackCandidates = electronCkfTrackCandidates.clone(
    src = "uncleanedOnlyElectronSeeds"
    )

from TrackingTools.GsfTracking.GsfElectronGsfFit_cff import *
uncleanedOnlyElectronGsfTracks = electronGsfTracks.clone(
    src = 'uncleanedOnlyElectronCkfTrackCandidates'
    )

uncleanedOnlyTrackingTask = cms.Task(uncleanedOnlyElectronSeeds,uncleanedOnlyElectronCkfTrackCandidates,uncleanedOnlyElectronGsfTracks)
uncleanedOnlyTracking = cms.Sequence(uncleanedOnlyTrackingTask)
#
# Conversions
#

from RecoEgamma.EgammaPhotonProducers.conversionTrackCandidates_cff import *
uncleanedOnlyConversionTrackCandidates = conversionTrackCandidates.clone(
    scHybridBarrelProducer  = "uncleanedOnlyCorrectedHybridSuperClusters",
    bcBarrelCollection      = "hybridSuperClusters:uncleanOnlyHybridSuperClusters",
    scIslandEndcapProducer  = "uncleanedOnlyCorrectedMulti5x5SuperClustersWithPreshower",
    bcEndcapCollection      = "multi5x5SuperClusters:uncleanOnlyMulti5x5EndcapBasicClusters"
    )

from RecoEgamma.EgammaPhotonProducers.ckfOutInTracksFromConversions_cfi import *
uncleanedOnlyCkfOutInTracksFromConversions = ckfOutInTracksFromConversions.clone(
    src           = "uncleanedOnlyConversionTrackCandidates:outInTracksFromConversions",
    producer      = 'uncleanedOnlyConversionTrackCandidates',
    ComponentName = 'uncleanedOnlyCkfOutInTracksFromConversions'
    )

from RecoEgamma.EgammaPhotonProducers.ckfInOutTracksFromConversions_cfi import *
uncleanedOnlyCkfInOutTracksFromConversions = ckfInOutTracksFromConversions.clone(
    src           = "uncleanedOnlyConversionTrackCandidates:inOutTracksFromConversions",
    producer      = 'uncleanedOnlyConversionTrackCandidates',
    ComponentName = 'uncleanedOnlyCkfInOutTracksFromConversions'
    )

uncleanedOnlyCkfTracksFromConversionsTask = cms.Task(uncleanedOnlyConversionTrackCandidates,uncleanedOnlyCkfOutInTracksFromConversions,uncleanedOnlyCkfInOutTracksFromConversions)
uncleanedOnlyCkfTracksFromConversions = cms.Sequence(uncleanedOnlyCkfTracksFromConversionsTask)

from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
uncleanedOnlyGeneralConversionTrackProducer = generalConversionTrackProducer.clone()

from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
uncleanedOnlyInOutConversionTrackProducer = inOutConversionTrackProducer.clone(
    TrackProducer = 'uncleanedOnlyCkfInOutTracksFromConversions'
    )

from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
uncleanedOnlyOutInConversionTrackProducer = outInConversionTrackProducer.clone(
    TrackProducer = 'uncleanedOnlyCkfOutInTracksFromConversions'
    )

from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
uncleanedOnlyGsfConversionTrackProducer = gsfConversionTrackProducer.clone(
    TrackProducer = 'uncleanedOnlyElectronGsfTracks'
    )

uncleanedOnlyConversionTrackProducersTask  = cms.Task(uncleanedOnlyGeneralConversionTrackProducer,uncleanedOnlyInOutConversionTrackProducer,uncleanedOnlyOutInConversionTrackProducer,uncleanedOnlyGsfConversionTrackProducer)
uncleanedOnlyConversionTrackProducers  = cms.Sequence(uncleanedOnlyConversionTrackProducersTask)

from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
uncleanedOnlyInOutOutInConversionTrackMerger = inOutOutInConversionTrackMerger.clone(
    TrackProducer2 = 'uncleanedOnlyOutInConversionTrackProducer',
    TrackProducer1 = 'uncleanedOnlyInOutConversionTrackProducer'
    )

from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
uncleanedOnlyGeneralInOutOutInConversionTrackMerger = generalInOutOutInConversionTrackMerger.clone(
    TrackProducer2 = 'uncleanedOnlyGeneralConversionTrackProducer',
    TrackProducer1 = 'uncleanedOnlyInOutOutInConversionTrackMerger'
    )

from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
uncleanedOnlyGsfGeneralInOutOutInConversionTrackMerger = gsfGeneralInOutOutInConversionTrackMerger.clone(
    TrackProducer2 = 'uncleanedOnlyGsfConversionTrackProducer',
    TrackProducer1 = 'uncleanedOnlyGeneralInOutOutInConversionTrackMerger'
    )

uncleanedOnlyConversionTrackMergersTask = cms.Task(uncleanedOnlyInOutOutInConversionTrackMerger,uncleanedOnlyGeneralInOutOutInConversionTrackMerger,uncleanedOnlyGsfGeneralInOutOutInConversionTrackMerger)
uncleanedOnlyConversionTrackMergers = cms.Sequence(uncleanedOnlyConversionTrackMergersTask)

from RecoEgamma.EgammaPhotonProducers.allConversions_cfi import *
uncleanedOnlyAllConversions = allConversions.clone(
    scBarrelProducer    = "uncleanedOnlyCorrectedHybridSuperClusters",
    bcBarrelCollection  = "hybridSuperClusters:uncleanOnlyHybridSuperClusters",
    scEndcapProducer    = "uncleanedOnlyCorrectedMulti5x5SuperClustersWithPreshower",
    bcEndcapCollection  = "multi5x5SuperClusters:uncleanOnlyMulti5x5EndcapBasicClusters",
    src                 = "uncleanedOnlyGsfGeneralInOutOutInConversionTrackMerger"
    )

uncleanedOnlyConversionsTask = cms.Task(uncleanedOnlyCkfTracksFromConversionsTask,uncleanedOnlyConversionTrackProducersTask,uncleanedOnlyConversionTrackMergersTask,uncleanedOnlyAllConversions)
uncleanedOnlyConversions = cms.Sequence(uncleanedOnlyConversionsTask)
#
# Particle Flow Tracking
#

from RecoParticleFlow.PFTracking.pfTrack_cfi import *
uncleanedOnlyPfTrack = pfTrack.clone(
    GsfTrackModuleLabel = "uncleanedOnlyElectronGsfTracks"
    )

from RecoParticleFlow.PFTracking.pfConversions_cfi import *
uncleanedOnlyPfConversions = pfConversions.clone(
    conversionCollection = "allConversions"
    )

from RecoParticleFlow.PFTracking.pfTrackElec_cfi import *
uncleanedOnlyPfTrackElec = pfTrackElec.clone(
    PFConversions       = "uncleanedOnlyPfConversions",
    GsfTrackModuleLabel = "uncleanedOnlyElectronGsfTracks",
    PFRecTrackLabel     = "uncleanedOnlyPfTrack"
    )

uncleanedOnlyPfTrackingTask = cms.Task(uncleanedOnlyPfTrack,uncleanedOnlyPfConversions,uncleanedOnlyPfTrackElec)
uncleanedOnlyPfTracking = cms.Sequence(uncleanedOnlyPfTrackingTask)

#
# Electrons
#

from RecoEgamma.EgammaElectronProducers.ecalDrivenGsfElectronCores_cfi import ecalDrivenGsfElectronCores
from RecoEgamma.EgammaElectronProducers.ecalDrivenGsfElectronCoresHGC_cff import ecalDrivenGsfElectronCoresHGC
uncleanedOnlyGsfElectronCores = ecalDrivenGsfElectronCores.clone(
    gsfTracks      = "uncleanedOnlyElectronGsfTracks",
    gsfPfRecTracks = "uncleanedOnlyPfTrackElec"
    )

from RecoEgamma.EgammaElectronProducers.gsfElectrons_cfi import *
uncleanedOnlyGsfElectrons = ecalDrivenGsfElectrons.clone(
    gsfPfRecTracksTag   = "uncleanedOnlyPfTrackElec",
    gsfElectronCoresTag = "uncleanedOnlyGsfElectronCores",
    seedsTag            = "uncleanedOnlyElectronSeeds"
    )

uncleanedOnlyElectronsTask = cms.Task(uncleanedOnlyGsfElectronCores,uncleanedOnlyGsfElectrons)
uncleanedOnlyElectrons = cms.Sequence(uncleanedOnlyElectronsTask)
#
# Whole Sequence
#

uncleanedOnlyElectronTask = cms.Task(uncleanedOnlyTrackingTask,uncleanedOnlyConversionsTask,uncleanedOnlyPfTrackingTask,uncleanedOnlyElectronsTask)
uncleanedOnlyElectronSequence = cms.Sequence(uncleanedOnlyElectronTask)
