
#import copy
#from PhysicsTools.PatAlgos.tools.helpers import *

#
# Tracking
#

from RecoEgamma.EgammaElectronProducers.ecalDrivenElectronSeeds_cff import *
uncleanedOnlyElectronSeeds = ecalDrivenElectronSeeds.clone(
    barrelSuperClusters = cms.InputTag("uncleanedOnlyCorrectedHybridSuperClusters"),
    endcapSuperClusters = cms.InputTag("uncleanedOnlyCorrectedMulti5x5SuperClustersWithPreshower")
    )

from TrackingTools.GsfTracking.CkfElectronCandidateMaker_cff import *
uncleanedOnlyElectronCkfTrackCandidates = electronCkfTrackCandidates.clone(
    src = cms.InputTag("uncleanedOnlyElectronSeeds")
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

from RecoEgamma.EgammaPhotonProducers.conversionTrackCandidates_cfi import *
uncleanedOnlyConversionTrackCandidates = conversionTrackCandidates.clone(
    scHybridBarrelProducer = cms.InputTag("uncleanedOnlyCorrectedHybridSuperClusters"),
    bcBarrelCollection  = cms.InputTag("hybridSuperClusters","uncleanOnlyHybridSuperClusters"),
    scIslandEndcapProducer  = cms.InputTag("uncleanedOnlyCorrectedMulti5x5SuperClustersWithPreshower"),
    bcEndcapCollection  = cms.InputTag("multi5x5SuperClusters","uncleanOnlyMulti5x5EndcapBasicClusters")
    )

from RecoEgamma.EgammaPhotonProducers.ckfOutInTracksFromConversions_cfi import *
uncleanedOnlyCkfOutInTracksFromConversions = ckfOutInTracksFromConversions.clone(
    src = cms.InputTag("uncleanedOnlyConversionTrackCandidates","outInTracksFromConversions"),
    producer = cms.string('uncleanedOnlyConversionTrackCandidates'),
    ComponentName = cms.string('uncleanedOnlyCkfOutInTracksFromConversions')
    )

from RecoEgamma.EgammaPhotonProducers.ckfInOutTracksFromConversions_cfi import *
uncleanedOnlyCkfInOutTracksFromConversions = ckfInOutTracksFromConversions.clone(
    src = cms.InputTag("uncleanedOnlyConversionTrackCandidates","inOutTracksFromConversions"),
    producer = cms.string('uncleanedOnlyConversionTrackCandidates'),
    ComponentName = cms.string('uncleanedOnlyCkfInOutTracksFromConversions')
    )

uncleanedOnlyCkfTracksFromConversionsTask = cms.Task(uncleanedOnlyConversionTrackCandidates,uncleanedOnlyCkfOutInTracksFromConversions,uncleanedOnlyCkfInOutTracksFromConversions)
uncleanedOnlyCkfTracksFromConversions = cms.Sequence(uncleanedOnlyCkfTracksFromConversionsTask)

from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
uncleanedOnlyGeneralConversionTrackProducer = generalConversionTrackProducer.clone()

from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
uncleanedOnlyInOutConversionTrackProducer = inOutConversionTrackProducer.clone(
    TrackProducer = cms.string('uncleanedOnlyCkfInOutTracksFromConversions')
    )

from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
uncleanedOnlyOutInConversionTrackProducer = outInConversionTrackProducer.clone(
    TrackProducer = cms.string('uncleanedOnlyCkfOutInTracksFromConversions')
    )

from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
uncleanedOnlyGsfConversionTrackProducer = gsfConversionTrackProducer.clone(
    TrackProducer = cms.string('uncleanedOnlyElectronGsfTracks')
    )

uncleanedOnlyConversionTrackProducersTask  = cms.Task(uncleanedOnlyGeneralConversionTrackProducer,uncleanedOnlyInOutConversionTrackProducer,uncleanedOnlyOutInConversionTrackProducer,uncleanedOnlyGsfConversionTrackProducer)
uncleanedOnlyConversionTrackProducers  = cms.Sequence(uncleanedOnlyConversionTrackProducersTask)

from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
uncleanedOnlyInOutOutInConversionTrackMerger = inOutOutInConversionTrackMerger.clone(
    TrackProducer2 = cms.InputTag('uncleanedOnlyOutInConversionTrackProducer'),
    TrackProducer1 = cms.InputTag('uncleanedOnlyInOutConversionTrackProducer')
    )

from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
uncleanedOnlyGeneralInOutOutInConversionTrackMerger = generalInOutOutInConversionTrackMerger.clone(
    TrackProducer2 = cms.InputTag('uncleanedOnlyGeneralConversionTrackProducer'),
    TrackProducer1 = cms.InputTag('uncleanedOnlyInOutOutInConversionTrackMerger')
    )

from RecoEgamma.EgammaPhotonProducers.conversionTrackSequence_cff import *
uncleanedOnlyGsfGeneralInOutOutInConversionTrackMerger = gsfGeneralInOutOutInConversionTrackMerger.clone(
    TrackProducer2 = cms.InputTag('uncleanedOnlyGsfConversionTrackProducer'),
    TrackProducer1 = cms.InputTag('uncleanedOnlyGeneralInOutOutInConversionTrackMerger')
    )

uncleanedOnlyConversionTrackMergersTask = cms.Task(uncleanedOnlyInOutOutInConversionTrackMerger,uncleanedOnlyGeneralInOutOutInConversionTrackMerger,uncleanedOnlyGsfGeneralInOutOutInConversionTrackMerger)
uncleanedOnlyConversionTrackMergers = cms.Sequence(uncleanedOnlyConversionTrackMergersTask)

from RecoEgamma.EgammaPhotonProducers.allConversions_cfi import *
uncleanedOnlyAllConversions = allConversions.clone(
    scBarrelProducer = cms.InputTag("uncleanedOnlyCorrectedHybridSuperClusters"),
    bcBarrelCollection  = cms.InputTag("hybridSuperClusters","uncleanOnlyHybridSuperClusters"),
    scEndcapProducer = cms.InputTag("uncleanedOnlyCorrectedMulti5x5SuperClustersWithPreshower"),
    bcEndcapCollection = cms.InputTag("multi5x5SuperClusters","uncleanOnlyMulti5x5EndcapBasicClusters"),
    src = cms.InputTag("uncleanedOnlyGsfGeneralInOutOutInConversionTrackMerger")
    )

uncleanedOnlyConversionsTask = cms.Task(uncleanedOnlyCkfTracksFromConversionsTask,uncleanedOnlyConversionTrackProducersTask,uncleanedOnlyConversionTrackMergersTask,uncleanedOnlyAllConversions)
uncleanedOnlyConversions = cms.Sequence(uncleanedOnlyConversionsTask)
#
# Particle Flow Tracking
#

from RecoParticleFlow.PFTracking.pfTrack_cfi import *
uncleanedOnlyPfTrack = pfTrack.clone(
    GsfTrackModuleLabel = cms.InputTag("uncleanedOnlyElectronGsfTracks")
    )

from RecoParticleFlow.PFTracking.pfConversions_cfi import *
uncleanedOnlyPfConversions = pfConversions.clone(
    conversionCollection = cms.InputTag("allConversions")
    )

from RecoParticleFlow.PFTracking.pfTrackElec_cfi import *
uncleanedOnlyPfTrackElec = pfTrackElec.clone(
    PFConversions = cms.InputTag("uncleanedOnlyPfConversions"),
    GsfTrackModuleLabel = cms.InputTag("uncleanedOnlyElectronGsfTracks"),
    PFRecTrackLabel = cms.InputTag("uncleanedOnlyPfTrack")
    )

uncleanedOnlyPfTrackingTask = cms.Task(uncleanedOnlyPfTrack,uncleanedOnlyPfConversions,uncleanedOnlyPfTrackElec)
uncleanedOnlyPfTracking = cms.Sequence(uncleanedOnlyPfTrackingTask)

#
# Electrons
#

from RecoEgamma.EgammaElectronProducers.ecalDrivenGsfElectronCores_cfi import ecalDrivenGsfElectronCores
from RecoEgamma.EgammaElectronProducers.ecalDrivenGsfElectronCoresFromMultiCl_cff import ecalDrivenGsfElectronCoresFromMultiCl
uncleanedOnlyGsfElectronCores = ecalDrivenGsfElectronCores.clone(
    gsfTracks = cms.InputTag("uncleanedOnlyElectronGsfTracks"),
    gsfPfRecTracks = cms.InputTag("uncleanedOnlyPfTrackElec")
    )

from RecoEgamma.EgammaElectronProducers.gsfElectrons_cfi import *
uncleanedOnlyGsfElectrons = ecalDrivenGsfElectrons.clone(
    gsfPfRecTracksTag = cms.InputTag("uncleanedOnlyPfTrackElec"),
    gsfElectronCoresTag = cms.InputTag("uncleanedOnlyGsfElectronCores"),
    seedsTag = cms.InputTag("uncleanedOnlyElectronSeeds")
    )

uncleanedOnlyElectronsTask = cms.Task(uncleanedOnlyGsfElectronCores,uncleanedOnlyGsfElectrons)
uncleanedOnlyElectrons = cms.Sequence(uncleanedOnlyElectronsTask)
#
# Whole Sequence
#

uncleanedOnlyElectronTask = cms.Task(uncleanedOnlyTrackingTask,uncleanedOnlyConversionsTask,uncleanedOnlyPfTrackingTask,uncleanedOnlyElectronsTask)
uncleanedOnlyElectronSequence = cms.Sequence(uncleanedOnlyElectronTask)
