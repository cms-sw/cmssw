import FWCore.ParameterSet.Config as cms

hltMtdTrackingRecHits = cms.EDProducer('MTDTrackingRecHitProducer',
                                       barrelClusters = cms.InputTag('hltMtdClusters', 'FTLBarrel'),
                                       endcapClusters = cms.InputTag('hltMtdClusters', 'FTLEndcap'),
                                       mightGet = cms.optional.untracked.vstring
                                       )
