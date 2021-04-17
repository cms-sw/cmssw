import FWCore.ParameterSet.Config as cms

import RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi
import RecoEgamma.EgammaPhotonProducers.conversionTrackMerger_cfi

# Conversion Track candidate producer 
from RecoEgamma.EgammaPhotonProducers.conversionTrackCandidates_cff import *
# Conversion Track producer  ( final fit )
from RecoEgamma.EgammaPhotonProducers.ckfOutInTracksFromConversions_cfi import *
from RecoEgamma.EgammaPhotonProducers.ckfInOutTracksFromConversions_cfi import *

ckfTracksFromConversionsTask = cms.Task(conversionTrackCandidates,ckfOutInTracksFromConversions,ckfInOutTracksFromConversions)
ckfTracksFromConversions = cms.Sequence(ckfTracksFromConversionsTask)

oldegConversionTrackCandidates = conversionTrackCandidates.clone(
    scHybridBarrelProducer = "correctedHybridSuperClusters",
    bcBarrelCollection     = "hybridSuperClusters:hybridBarrelBasicClusters",
    scIslandEndcapProducer = "correctedMulti5x5SuperClustersWithPreshower",
    bcEndcapCollection     = "multi5x5SuperClusters:multi5x5EndcapBasicClusters"
)
ckfOutInTracksFromOldEGConversions = ckfOutInTracksFromConversions.clone(
    src           = 'oldegConversionTrackCandidates:outInTracksFromConversions',
    producer      = 'oldegConversionTrackCandidates',
    ComponentName = 'ckfOutInTracksFromOldEGConversions'
)
ckfInOutTracksFromOldEGConversions = ckfInOutTracksFromConversions.clone(
    src           = 'oldegConversionTrackCandidates:inOutTracksFromConversions',
    producer      = 'oldegConversionTrackCandidates',
    ComponentName = 'ckfInOutTracksFromOldEGConversions'
)
ckfTracksFromOldEGConversionsTask = cms.Task(oldegConversionTrackCandidates,ckfOutInTracksFromOldEGConversions,ckfInOutTracksFromOldEGConversions)
ckfTracksFromOldEGConversions = cms.Sequence(ckfTracksFromOldEGConversionsTask)
#producer from general tracks collection, set tracker only, merged arbitrated, merged arbitrated ecal/general flags
generalConversionTrackProducer = RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi.conversionTrackProducer.clone(
    TrackProducer  = 'generalTracks',
    setTrackerOnly = True,
    setArbitratedMergedEcalGeneral = True,
)

#fastSim
from Configuration.Eras.Modifier_fastSim_cff import fastSim
# the conversion producer depends on trajectories
# so we feed it with the 'before mixing' track collection
generalConversionTrackProducerTmp = generalConversionTrackProducer.clone(
    TrackProducer = 'generalTracksBeforeMixing')

# then we need to fix the track references, so that they point to the final track collection, after mixing
import FastSimulation.Tracking.ConversionTrackRefFix_cfi
_fastSim_conversionTrackRefFix = FastSimulation.Tracking.ConversionTrackRefFix_cfi.fixedConversionTracks.clone(
                 src = "generalConversionTrackProducerTmp")
fastSim.toReplaceWith(generalConversionTrackProducer,
                      _fastSim_conversionTrackRefFix)


#producer from conversionStep tracks collection, set tracker only, merged arbitrated, merged arbitrated ecal/general flags
conversionStepConversionTrackProducer = RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi.conversionTrackProducer.clone(
    TrackProducer  = 'conversionStepTracks',
    setTrackerOnly = True,
    setArbitratedMergedEcalGeneral = True,
)


#producer from inout ecal seeded tracks, set arbitratedecalseeded, mergedarbitratedecalgeneral and mergedarbitrated flags
inOutConversionTrackProducer = RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi.conversionTrackProducer.clone(
    TrackProducer           = 'ckfInOutTracksFromConversions',
    setArbitratedEcalSeeded = True,
    setArbitratedMergedEcalGeneral = True,
)

#producer from outin ecal seeded tracks, set arbitratedecalseeded, mergedarbitratedecalgeneral and mergedarbitrated flags
outInConversionTrackProducer = RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi.conversionTrackProducer.clone(
    TrackProducer           = 'ckfOutInTracksFromConversions',
    setArbitratedEcalSeeded = True,
    setArbitratedMergedEcalGeneral = True,
)

#producer from gsf tracks, set only mergedarbitrated flag (default behaviour)
gsfConversionTrackProducer = RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi.conversionTrackProducer.clone(
    TrackProducer        = 'electronGsfTracks',
    filterOnConvTrackHyp = False,
)

conversionTrackProducersTask = cms.Task(generalConversionTrackProducer,conversionStepConversionTrackProducer,inOutConversionTrackProducer,outInConversionTrackProducer,gsfConversionTrackProducer)
conversionTrackProducers = cms.Sequence(conversionTrackProducersTask)

inOutOldEGConversionTrackProducer = inOutConversionTrackProducer.clone(
    TrackProducer = 'ckfInOutTracksFromOldEGConversions'
)
outInOldEGConversionTrackProducer = outInConversionTrackProducer.clone(
    TrackProducer = 'ckfOutInTracksFromOldEGConversions'
)
oldegConversionTrackProducersTask = cms.Task(inOutOldEGConversionTrackProducer,outInOldEGConversionTrackProducer)
oldegConversionTrackProducers = cms.Sequence(oldegConversionTrackProducersTask)
#merge generalTracks and conversionStepTracks collections, with arbitration by nhits then chi^2/ndof for ecalseededarbitrated, mergedarbitratedecalgeneral and mergedarbitrated flags
generalConversionStepConversionTrackMerger = RecoEgamma.EgammaPhotonProducers.conversionTrackMerger_cfi.conversionTrackMerger.clone(
    TrackProducer1 = 'generalConversionTrackProducer',
    TrackProducer2 = 'conversionStepConversionTrackProducer',
    #prefer collection settings:
    #-1: propagate output/flag from both input collections
    # 0: propagate output/flag from neither input collection
    # 1: arbitrate output/flag (remove duplicates by shared hits), give precedence to first input collection
    # 2: arbitrate output/flag (remove duplicates by shared hits), give precedence to second input collection
    # 3: arbitrate output/flag (remove duplicates by shared hits), arbitration first by number of hits, second by chisq/ndof  
    arbitratedMergedPreferCollection            = 3,
    arbitratedMergedEcalGeneralPreferCollection = 3,        
)

#merge two ecal-seeded collections, with arbitration by nhits then chi^2/ndof for ecalseededarbitrated, mergedarbitratedecalgeneral and mergedarbitrated flags
inOutOutInConversionTrackMerger = RecoEgamma.EgammaPhotonProducers.conversionTrackMerger_cfi.conversionTrackMerger.clone(
    TrackProducer1 = 'inOutConversionTrackProducer',
    TrackProducer2 = 'outInConversionTrackProducer',
    #prefer collection settings:
    #-1: propagate output/flag from both input collections
    # 0: propagate output/flag from neither input collection
    # 1: arbitrate output/flag (remove duplicates by shared hits), give precedence to first input collection
    # 2: arbitrate output/flag (remove duplicates by shared hits), give precedence to second input collection
    # 3: arbitrate output/flag (remove duplicates by shared hits), arbitration first by number of hits, second by chisq/ndof  
    arbitratedEcalSeededPreferCollection        = 3,    
    arbitratedMergedPreferCollection            = 3,
    arbitratedMergedEcalGeneralPreferCollection = 3,        
)

#merge ecalseeded collections with collection from general tracks
#trackeronly flag is forwarded from the generaltrack-based collections
#ecalseeded flag is forwarded from the ecal seeded collection
#arbitratedmerged flag is set based on shared hit matching, arbitration by nhits then chi^2/ndof
#arbitratedmergedecalgeneral flag is set based on shared hit matching, precedence given to generalTracks
generalInOutOutInConversionTrackMerger = RecoEgamma.EgammaPhotonProducers.conversionTrackMerger_cfi.conversionTrackMerger.clone(
    TrackProducer1 = 'inOutOutInConversionTrackMerger',
    TrackProducer2 = 'generalConversionStepConversionTrackMerger',
    arbitratedMergedPreferCollection            = 3,
    arbitratedMergedEcalGeneralPreferCollection = 2,        
)

#merge the result of the above with the collection from gsf tracks
#trackeronly, arbitratedmergedecalgeneral, and mergedecal flags are forwarded
#arbitratedmerged flag set based on overlap removal by shared hits, with precedence given to gsf tracks
gsfGeneralInOutOutInConversionTrackMerger = RecoEgamma.EgammaPhotonProducers.conversionTrackMerger_cfi.conversionTrackMerger.clone(
    TrackProducer1 = 'generalInOutOutInConversionTrackMerger',
    TrackProducer2 = 'gsfConversionTrackProducer',
    arbitratedMergedPreferCollection = 2,
)

#final output collection contains combination of generaltracks, ecal seeded tracks and gsf tracks, with overlaps removed by shared hits
#precedence is given first to gsf tracks, then to the combination of ecal seeded and general tracks
#overlaps between the ecal seeded track collections and between ecal seeded and general tracks are arbitrated first by nhits then by chi^2/dof
#(logic and much of the code is adapted from FinalTrackSelectors)

conversionTrackMergersTask = cms.Task(inOutOutInConversionTrackMerger,generalConversionStepConversionTrackMerger,generalInOutOutInConversionTrackMerger,gsfGeneralInOutOutInConversionTrackMerger)
conversionTrackMergers = cms.Sequence(conversionTrackMergersTask)

inOutOutInOldEGConversionTrackMerger = inOutOutInConversionTrackMerger.clone(
    TrackProducer1 = 'inOutOldEGConversionTrackProducer',
    TrackProducer2 = 'outInOldEGConversionTrackProducer'
)
generalInOutOutInOldEGConversionTrackMerger = generalInOutOutInConversionTrackMerger.clone(
    TrackProducer1 = 'inOutOutInOldEGConversionTrackMerger'
)
gsfGeneralInOutOutInOldEGConversionTrackMerger = gsfGeneralInOutOutInConversionTrackMerger.clone(
    TrackProducer1 = 'generalInOutOutInOldEGConversionTrackMerger'
)
oldegConversionTrackMergersTask = cms.Task(inOutOutInOldEGConversionTrackMerger,generalInOutOutInOldEGConversionTrackMerger,gsfGeneralInOutOutInOldEGConversionTrackMerger)
oldegConversionTrackMergers = cms.Sequence(oldegConversionTrackMergersTask)


conversionTrackTask = cms.Task(ckfTracksFromConversionsTask,conversionTrackProducersTask,conversionTrackMergersTask)
conversionTrackSequence = cms.Sequence(conversionTrackTask)
#merge the general tracks with the collection from gsf tracks
#arbitratedmerged flag set based on overlap removal by shared hits, with precedence given to gsf tracks
gsfGeneralConversionTrackMerger = RecoEgamma.EgammaPhotonProducers.conversionTrackMerger_cfi.conversionTrackMerger.clone(
    TrackProducer1 = 'generalConversionTrackProducer',
    TrackProducer2 = 'gsfConversionTrackProducer',
    arbitratedMergedPreferCollection = 2,
)

#special sequence for fastsim which skips the ecal-seeded and conversionStep tracks for now
conversionTrackTaskNoEcalSeeded = cms.Task(generalConversionTrackProducer,gsfConversionTrackProducer,gsfGeneralConversionTrackMerger)
conversionTrackSequenceNoEcalSeeded = cms.Sequence(conversionTrackTaskNoEcalSeeded)

_fastSim_conversionTrackTaskNoEcalSeeded = conversionTrackTaskNoEcalSeeded.copy()
_fastSim_conversionTrackTaskNoEcalSeeded.replace(generalConversionTrackProducer,cms.Task(generalConversionTrackProducerTmp,generalConversionTrackProducer))
fastSim.toReplaceWith(conversionTrackTaskNoEcalSeeded,_fastSim_conversionTrackTaskNoEcalSeeded)
