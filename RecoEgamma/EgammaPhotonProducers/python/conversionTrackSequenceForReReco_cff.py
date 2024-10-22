import FWCore.ParameterSet.Config as cms

import RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi
import RecoEgamma.EgammaPhotonProducers.conversionTrackMerger_cfi

# Conversion Track candidate producer 
from RecoEgamma.EgammaPhotonProducers.conversionTrackCandidates_cff import *
# Conversion Track producer  ( final fit )
from RecoEgamma.EgammaPhotonProducers.ckfOutInTracksFromConversions_cfi import *
from RecoEgamma.EgammaPhotonProducers.ckfInOutTracksFromConversions_cfi import *
ckfTracksFromConversionsReRecoTask = cms.Task(conversionTrackCandidates,
                                              ckfOutInTracksFromConversions,
                                              ckfInOutTracksFromConversions)

#producer from general tracks collection, set tracker only and merged arbitrated flag
generalConversionTrackProducerReReco = RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi.conversionTrackProducer.clone(
    TrackProducer  = 'generalTracks',
    setTrackerOnly = True,
    useTrajectory  = False,
)

#producer from conversionStep tracks collection, set tracker only, merged arbitrated, merged arbitrated ecal/general flags
conversionStepConversionTrackProducerReReco = RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi.conversionTrackProducer.clone(
    TrackProducer  = 'conversionStepTracks',
    setTrackerOnly = True,
    setArbitratedMergedEcalGeneral = True,
    useTrajectory  = False,
)

#producer from inout ecal seeded tracks, set arbitratedecalseeded and mergedarbitrated flags
inOutConversionTrackProducerReReco = RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi.conversionTrackProducer.clone(
    TrackProducer = 'ckfInOutTracksFromConversions',
    setArbitratedEcalSeeded = True,
    useTrajectory = False,
)

#producer from outin ecal seeded tracks, set arbitratedecalseeded and mergedarbitrated flags
outInConversionTrackProducerReReco = RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi.conversionTrackProducer.clone(
    TrackProducer = 'ckfOutInTracksFromConversions',
    setArbitratedEcalSeeded = True,
    useTrajectory = False,
)

#producer from gsf tracks, set only mergedarbitrated flag (default behaviour)
gsfConversionTrackProducerReReco = RecoEgamma.EgammaPhotonProducers.conversionTrackProducer_cfi.conversionTrackProducer.clone(
    TrackProducer = 'electronGsfTracks',
    useTrajectory = False,
)

conversionTrackProducersReRecoTask = cms.Task(generalConversionTrackProducerReReco,
                                              conversionStepConversionTrackProducerReReco,
                                              inOutConversionTrackProducerReReco,
                                              outInConversionTrackProducerReReco,
                                              gsfConversionTrackProducerReReco)

#merge generalTracks and conversionStepTracks collections, with arbitration by nhits then chi^2/ndof for ecalseededarbitrated, mergedarbitratedecalgeneral and mergedarbitrated flags
generalConversionStepConversionTrackMergerReReco = RecoEgamma.EgammaPhotonProducers.conversionTrackMerger_cfi.conversionTrackMerger.clone(
    TrackProducer1 = 'generalConversionTrackProducerReReco',
    TrackProducer2 = 'conversionStepConversionTrackProducerReReco',
    #prefer collection settings:
    #-1: propagate output/flag from both input collections
    # 0: propagate output/flag from neither input collection
    # 1: arbitrate output/flag (remove duplicates by shared hits), give precedence to first input collection
    # 2: arbitrate output/flag (remove duplicates by shared hits), give precedence to second input collection
    # 3: arbitrate output/flag (remove duplicates by shared hits), arbitration first by number of hits, second by chisq/ndof  
    arbitratedMergedPreferCollection = 3,
    arbitratedMergedEcalGeneralPreferCollection = 3,
)

#merge two ecal-seeded collections, with arbitration by nhits then chi^2/ndof for both ecalseededarbitrated and mergedarbitrated flags
inOutOutInConversionTrackMergerReReco = RecoEgamma.EgammaPhotonProducers.conversionTrackMerger_cfi.conversionTrackMerger.clone(
    TrackProducer1 = 'inOutConversionTrackProducerReReco',
    TrackProducer2 = 'outInConversionTrackProducerReReco',
    #prefer collection settings:
    #-1: propagate output/flag from both input collections
    # 0: propagate output/flag from neither input collection
    # 1: arbitrate output/flag (remove duplicates by shared hits), give precedence to first input collection
    # 2: arbitrate output/flag (remove duplicates by shared hits), give precedence to second input collection
    # 3: arbitrate output/flag (remove duplicates by shared hits), arbitration first by number of hits, second by chisq/ndof  
    arbitratedEcalSeededPreferCollection = 3,    
    arbitratedMergedPreferCollection = 3,
)

#merge ecalseeded collections with collection from general tracks
#trackeronly flag is forwarded from the generaltrack-based collections
#ecalseeded flag is forwarded from the ecal seeded collection
#arbitratedmerged flag is set based on shared hit matching, arbitration by nhits then chi^2/ndof
generalInOutOutInConversionTrackMergerReReco = RecoEgamma.EgammaPhotonProducers.conversionTrackMerger_cfi.conversionTrackMerger.clone(
    TrackProducer1 = 'inOutOutInConversionTrackMergerReReco',
    TrackProducer2 = 'generalConversionStepConversionTrackMergerReReco',
    arbitratedMergedPreferCollection = 3,
)

#merge the result of the above with the collection from gsf tracks
#trackeronly and mergedecal flags are forwarded
#arbitratedmerged flag set based on overlap removal by shared hits, with precedence given to gsf tracks
gsfGeneralInOutOutInConversionTrackMergerReReco = RecoEgamma.EgammaPhotonProducers.conversionTrackMerger_cfi.conversionTrackMerger.clone(
    TrackProducer1 = 'generalInOutOutInConversionTrackMergerReReco',
    TrackProducer2 = 'gsfConversionTrackProducerReReco',
    arbitratedMergedPreferCollection = 2,
)

#final output collection contains combination of generaltracks, ecal seeded tracks and gsf tracks, with overlaps removed by shared hits
#precedence is given first to gsf tracks, then to the combination of ecal seeded and general tracks
#overlaps between the ecal seeded track collections and between ecal seeded and general tracks are arbitrated first by nhits then by chi^2/dof
#(logic and much of the code is adapted from FinalTrackSelectors)

conversionTrackMergersReRecoTask = cms.Task(inOutOutInConversionTrackMergerReReco,
                                            generalConversionStepConversionTrackMergerReReco,
                                            generalInOutOutInConversionTrackMergerReReco,
                                            gsfGeneralInOutOutInConversionTrackMergerReReco)

conversionTrackTaskForReReco = cms.Task(ckfTracksFromConversionsReRecoTask,
                                        conversionTrackProducersReRecoTask,
                                        conversionTrackMergersReRecoTask)
conversionTrackSequenceForReReco = cms.Sequence(conversionTrackTaskForReReco)

#merge the general tracks with the collection from gsf tracks
#arbitratedmerged flag set based on overlap removal by shared hits, with precedence given to gsf tracks
gsfGeneralConversionTrackMergerReReco = RecoEgamma.EgammaPhotonProducers.conversionTrackMerger_cfi.conversionTrackMerger.clone(
    TrackProducer1 = 'generalConversionTrackProducerReReco',
    TrackProducer2 = 'gsfConversionTrackProducerReReco',
    arbitratedMergedPreferCollection = 2,
)

#conversionTrackSequenceNoEcalSeeded = cms.Sequence(generalConversionTrackProducer*gsfConversionTrackProducer*gsfGeneralConversionTrackMerger)
