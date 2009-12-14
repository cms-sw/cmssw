import FWCore.ParameterSet.Config as cms

from CalibTracker.SiStripCommon.ShallowEventDataProducer_cfi import *
from CalibTracker.SiStripCommon.ShallowDigisProducer_cfi import *
from CalibTracker.SiStripCommon.ShallowClustersProducer_cfi import *
from CalibTracker.SiStripCommon.ShallowTrackClustersProducer_cfi import *
from CalibTracker.SiStripCommon.ShallowRechitClustersProducer_cfi import *
from CalibTracker.SiStripCommon.ShallowTracksProducer_cfi import *

from RecoVertex.BeamSpotProducer.BeamSpot_cff import *
from RecoTracker.TrackProducer.TrackRefitters_cff import *

bigNtupleTrackCollectionTag = cms.InputTag("bigNtupleTracksRefit")
bigNtupleClusterCollectionTag = cms.InputTag("siStripClusters")

bigNtupleTracksRefit =  RecoTracker.TrackProducer.TrackRefitter_cfi.TrackRefitter.clone(src = bigNtupleTrackCollectionTag)


bigNtupleEventRun        = shallowEventRun.clone()
bigNtupleDigis           = shallowDigis.clone()
bigNtupleClusters        = shallowClusters.clone(Clusters=bigNtupleClusterCollectionTag)
bigNtupleRecHits         = shallowRechitClusters.clone(Clusters=bigNtupleClusterCollectionTag)
bigNtupleTrackClusters   = shallowTrackClusters.clone(Tracks = bigNtupleTrackCollectionTag,Clusters=bigNtupleClusterCollectionTag)
bigNtupleTracks          = shallowTracks.clone(Tracks = bigNtupleTrackCollectionTag)
    

bigShallowTree = cms.EDAnalyzer("ShallowTree",
                                outputCommands = cms.untracked.vstring(
    'drop *',
    'keep *_bigNtupleEventRun_*_*',
    'keep *_bigNtupleDigis_*_*',
    'keep *_bigNtupleClusters_*_*' ,
    'keep *_bigNtupleRechits_*_*',
    'keep *_bigNtupleTracks_*_*',
        'keep *_bigNtupleTrackClusters_*_*'
    )
                                )


theBigNtuple = cms.Sequence( (siStripMatchedRecHits + offlineBeamSpot + bigNtupleTracksRefit)
                             * (bigNtupleEventRun +
                                #bigNtupleDigis +
                                bigNtupleClusters +
                                bigNtupleRecHits +
                                bigNtupleTracks +
                                bigNtupleTrackClusters
                                )
                             *
                             bigShallowTree
                             )

