import FWCore.ParameterSet.Config as cms

import RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi
import RecoEgamma.EgammaPhotonProducers.conversionTrackMerger_cfi

# Conversion Track candidate producer 
from RecoEgamma.EgammaPhotonProducers.conversionTrackCandidates_cfi import *
# Conversion Track producer  ( final fit )
from RecoEgamma.EgammaPhotonProducers.ckfOutInTracksFromConversions_cfi import *
from RecoEgamma.EgammaPhotonProducers.ckfInOutTracksFromConversions_cfi import *

ckfTracksFromConversions = cms.Sequence(conversionTrackCandidates*ckfOutInTracksFromConversions*ckfInOutTracksFromConversions)

oldegConversionTrackCandidates = conversionTrackCandidates.clone()
oldegConversionTrackCandidates.scHybridBarrelProducer = cms.InputTag("correctedHybridSuperClusters")
oldegConversionTrackCandidates.bcBarrelCollection = cms.InputTag("hybridSuperClusters","hybridBarrelBasicClusters")
oldegConversionTrackCandidates.scIslandEndcapProducer = cms.InputTag("correctedMulti5x5SuperClustersWithPreshower")
oldegConversionTrackCandidates.bcEndcapCollection = cms.InputTag("multi5x5SuperClusters","multi5x5EndcapBasicClusters")

ckfOutInTracksFromOldEGConversions = ckfOutInTracksFromConversions.clone()
ckfOutInTracksFromOldEGConversions.src = cms.InputTag('oldegConversionTrackCandidates','outInTracksFromConversions')
ckfOutInTracksFromOldEGConversions.producer = cms.string('oldegConversionTrackCandidates')
ckfOutInTracksFromOldEGConversions.ComponentName = cms.string('ckfOutInTracksFromOldEGConversions')

ckfInOutTracksFromOldEGConversions = ckfInOutTracksFromConversions.clone()
ckfInOutTracksFromOldEGConversions.src = cms.InputTag('oldegConversionTrackCandidates','inOutTracksFromConversions')
ckfInOutTracksFromOldEGConversions.producer = cms.string('oldegConversionTrackCandidates')
ckfInOutTracksFromOldEGConversions.ComponentName = cms.string('ckfInOutTracksFromOldEGConversions')

ckfTracksFromOldEGConversions = cms.Sequence(oldegConversionTrackCandidates*ckfOutInTracksFromOldEGConversions*ckfInOutTracksFromOldEGConversions)

#producer from general tracks collection, set tracker only, merged arbitrated, merged arbitrated ecal/general flags
generalConversionTrackProducer = RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi.conversionTrackProducer.clone(
    TrackProducer = cms.string('generalTracks'),
    setTrackerOnly = cms.bool(True),
    setArbitratedMergedEcalGeneral = cms.bool(True),
)

#producer from conversionStep tracks collection, set tracker only, merged arbitrated, merged arbitrated ecal/general flags
conversionStepConversionTrackProducer = RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi.conversionTrackProducer.clone(
    TrackProducer = cms.string('conversionStepTracks'),
    setTrackerOnly = cms.bool(True),
    setArbitratedMergedEcalGeneral = cms.bool(True),
)


#producer from inout ecal seeded tracks, set arbitratedecalseeded, mergedarbitratedecalgeneral and mergedarbitrated flags
inOutConversionTrackProducer = RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi.conversionTrackProducer.clone(
    TrackProducer = cms.string('ckfInOutTracksFromConversions'),
    setArbitratedEcalSeeded = cms.bool(True),
    setArbitratedMergedEcalGeneral = cms.bool(True),    
)

#producer from outin ecal seeded tracks, set arbitratedecalseeded, mergedarbitratedecalgeneral and mergedarbitrated flags
outInConversionTrackProducer = RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi.conversionTrackProducer.clone(
    TrackProducer = cms.string('ckfOutInTracksFromConversions'),
    setArbitratedEcalSeeded = cms.bool(True),
    setArbitratedMergedEcalGeneral = cms.bool(True),    
)

#producer from gsf tracks, set only mergedarbitrated flag (default behaviour)
gsfConversionTrackProducer = RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi.conversionTrackProducer.clone(
    TrackProducer = cms.string('electronGsfTracks'),
    filterOnConvTrackHyp = cms.bool(False),
)

conversionTrackProducers = cms.Sequence(generalConversionTrackProducer*conversionStepConversionTrackProducer*inOutConversionTrackProducer*outInConversionTrackProducer*gsfConversionTrackProducer)

inOutOldEGConversionTrackProducer = inOutConversionTrackProducer.clone()
inOutOldEGConversionTrackProducer.TrackProducer = cms.string('ckfInOutTracksFromOldEGConversions')
outInOldEGConversionTrackProducer = outInConversionTrackProducer.clone()
outInOldEGConversionTrackProducer.TrackProducer = cms.string('ckfOutInTracksFromOldEGConversions')

oldegConversionTrackProducers = cms.Sequence(inOutOldEGConversionTrackProducer*outInOldEGConversionTrackProducer)

#merge generalTracks and conversionStepTracks collections, with arbitration by nhits then chi^2/ndof for ecalseededarbitrated, mergedarbitratedecalgeneral and mergedarbitrated flags
generalConversionStepConversionTrackMerger = RecoEgamma.EgammaPhotonProducers.conversionTrackMerger_cfi.conversionTrackMerger.clone(
    TrackProducer1 = cms.InputTag('generalConversionTrackProducer'),
    TrackProducer2 = cms.InputTag('conversionStepConversionTrackProducer'),
    #prefer collection settings:
    #-1: propagate output/flag from both input collections
    # 0: propagate output/flag from neither input collection
    # 1: arbitrate output/flag (remove duplicates by shared hits), give precedence to first input collection
    # 2: arbitrate output/flag (remove duplicates by shared hits), give precedence to second input collection
    # 3: arbitrate output/flag (remove duplicates by shared hits), arbitration first by number of hits, second by chisq/ndof  
    arbitratedMergedPreferCollection = cms.int32(3),
    arbitratedMergedEcalGeneralPreferCollection = cms.int32(3),        
)

#merge two ecal-seeded collections, with arbitration by nhits then chi^2/ndof for ecalseededarbitrated, mergedarbitratedecalgeneral and mergedarbitrated flags
inOutOutInConversionTrackMerger = RecoEgamma.EgammaPhotonProducers.conversionTrackMerger_cfi.conversionTrackMerger.clone(
    TrackProducer1 = cms.InputTag('inOutConversionTrackProducer'),
    TrackProducer2 = cms.InputTag('outInConversionTrackProducer'),
    #prefer collection settings:
    #-1: propagate output/flag from both input collections
    # 0: propagate output/flag from neither input collection
    # 1: arbitrate output/flag (remove duplicates by shared hits), give precedence to first input collection
    # 2: arbitrate output/flag (remove duplicates by shared hits), give precedence to second input collection
    # 3: arbitrate output/flag (remove duplicates by shared hits), arbitration first by number of hits, second by chisq/ndof  
    arbitratedEcalSeededPreferCollection = cms.int32(3),    
    arbitratedMergedPreferCollection = cms.int32(3),
    arbitratedMergedEcalGeneralPreferCollection = cms.int32(3),        
)

#merge ecalseeded collections with collection from general tracks
#trackeronly flag is forwarded from the generaltrack-based collections
#ecalseeded flag is forwarded from the ecal seeded collection
#arbitratedmerged flag is set based on shared hit matching, arbitration by nhits then chi^2/ndof
#arbitratedmergedecalgeneral flag is set based on shared hit matching, precedence given to generalTracks
generalInOutOutInConversionTrackMerger = RecoEgamma.EgammaPhotonProducers.conversionTrackMerger_cfi.conversionTrackMerger.clone(
    TrackProducer1 = cms.InputTag('inOutOutInConversionTrackMerger'),
    TrackProducer2 = cms.InputTag('generalConversionStepConversionTrackMerger'),
    arbitratedMergedPreferCollection = cms.int32(3),
    arbitratedMergedEcalGeneralPreferCollection = cms.int32(2),        
)

#merge the result of the above with the collection from gsf tracks
#trackeronly, arbitratedmergedecalgeneral, and mergedecal flags are forwarded
#arbitratedmerged flag set based on overlap removal by shared hits, with precedence given to gsf tracks
gsfGeneralInOutOutInConversionTrackMerger = RecoEgamma.EgammaPhotonProducers.conversionTrackMerger_cfi.conversionTrackMerger.clone(
    TrackProducer1 = cms.InputTag('generalInOutOutInConversionTrackMerger'),
    TrackProducer2 = cms.InputTag('gsfConversionTrackProducer'),
    arbitratedMergedPreferCollection = cms.int32(2),
)

#final output collection contains combination of generaltracks, ecal seeded tracks and gsf tracks, with overlaps removed by shared hits
#precedence is given first to gsf tracks, then to the combination of ecal seeded and general tracks
#overlaps between the ecal seeded track collections and between ecal seeded and general tracks are arbitrated first by nhits then by chi^2/dof
#(logic and much of the code is adapted from FinalTrackSelectors)

conversionTrackMergers = cms.Sequence(inOutOutInConversionTrackMerger*generalConversionStepConversionTrackMerger*generalInOutOutInConversionTrackMerger*gsfGeneralInOutOutInConversionTrackMerger)

inOutOutInOldEGConversionTrackMerger = inOutOutInConversionTrackMerger.clone()
inOutOutInOldEGConversionTrackMerger.TrackProducer1 = cms.InputTag('inOutOldEGConversionTrackProducer')
inOutOutInOldEGConversionTrackMerger.TrackProducer2 = cms.InputTag('outInOldEGConversionTrackProducer')

generalInOutOutInOldEGConversionTrackMerger = generalInOutOutInConversionTrackMerger.clone()
generalInOutOutInOldEGConversionTrackMerger.TrackProducer1 = cms.InputTag('inOutOutInOldEGConversionTrackMerger')

gsfGeneralInOutOutInOldEGConversionTrackMerger = gsfGeneralInOutOutInConversionTrackMerger.clone()
gsfGeneralInOutOutInOldEGConversionTrackMerger.TrackProducer1 = cms.InputTag('generalInOutOutInOldEGConversionTrackMerger')

oldegConversionTrackMergers = cms.Sequence(inOutOutInOldEGConversionTrackMerger*generalInOutOutInOldEGConversionTrackMerger*gsfGeneralInOutOutInOldEGConversionTrackMerger)

conversionTrackSequence = cms.Sequence(ckfTracksFromConversions*conversionTrackProducers*conversionTrackMergers)

#merge the general tracks with the collection from gsf tracks
#arbitratedmerged flag set based on overlap removal by shared hits, with precedence given to gsf tracks
gsfGeneralConversionTrackMerger = RecoEgamma.EgammaPhotonProducers.conversionTrackMerger_cfi.conversionTrackMerger.clone(
    TrackProducer1 = cms.InputTag('generalConversionTrackProducer'),
    TrackProducer2 = cms.InputTag('gsfConversionTrackProducer'),
    arbitratedMergedPreferCollection = cms.int32(2),
)

#special sequence for fastsim which skips the ecal-seeded and conversionStep tracks for now
conversionTrackSequenceNoEcalSeeded = cms.Sequence(generalConversionTrackProducer*gsfConversionTrackProducer*gsfGeneralConversionTrackMerger)
