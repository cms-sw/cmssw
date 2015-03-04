import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.DataMixerPreMix_cff import *
mixData.TrackerMergeType = "tracks"
mixData.GeneralTrackDigiCollectionDM = cms.string("generalTracks")
mixData.GeneralTrackLabelSig = cms.InputTag("generalTracksBeforePreMixing")
mixData.GeneralTrackPileInputTag = cms.InputTag("generalTracksBeforeMixing")
mixData.hitProducer = cms.InputTag("famosSimHits")
