import FWCore.ParameterSet.Config as cms

#
#
# Conversion Track candidate producer 
from RecoEgamma.EgammaPhotonProducers.conversionTrackCandidates_cff import *
# Conversion Track producer  ( final fit )
from RecoEgamma.EgammaPhotonProducers.ckfOutInTracksFromConversions_cfi import *
from RecoEgamma.EgammaPhotonProducers.ckfInOutTracksFromConversions_cfi import *
ckfTracksFromConversionsTask = cms.Task(conversionTrackCandidates,
                                        ckfOutInTracksFromConversions,
                                        ckfInOutTracksFromConversions)
ckfTracksFromConversions = cms.Sequence(ckfTracksFromConversionsTask)

mustacheConversionTrackCandidates = conversionTrackCandidates.clone(
    scHybridBarrelProducer = 'particleFlowSuperClusterECAL:particleFlowSuperClusterECALBarrel',
    bcBarrelCollection     = 'particleFlowSuperClusterECAL:particleFlowBasicClusterECALBarrel',
    scIslandEndcapProducer = 'particleFlowSuperClusterECAL:particleFlowSuperClusterECALEndcapWithPreshower',
    bcEndcapCollection     = 'particleFlowSuperClusterECAL:particleFlowBasicClusterECALEndcap'
)
ckfOutInTracksFromMustacheConversions = ckfOutInTracksFromConversions.clone(
    src           = 'mustacheConversionTrackCandidates:outInTracksFromConversions',
    producer      = 'mustacheConversionTrackCandidates',
    ComponentName = 'ckfOutInTracksFromMustacheConversions',
)
ckfInOutTracksFromMustacheConversions = ckfInOutTracksFromConversions.clone(
    src           = 'mustacheConversionTrackCandidates:inOutTracksFromConversions',
    producer      = 'mustacheConversionTrackCandidates',
    ComponentName = 'ckfInOutTracksFromMustacheConversions',
)
ckfTracksFromMustacheConversionsTask = cms.Task(mustacheConversionTrackCandidates,
                                                ckfOutInTracksFromMustacheConversions,
                                                ckfInOutTracksFromMustacheConversions)
ckfTracksFromMustacheConversions = cms.Sequence(ckfTracksFromMustacheConversionsTask)
