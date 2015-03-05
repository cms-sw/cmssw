import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.DataMixerPreMix_cff import *
mixData.TrackerMergeType = "tracks"
mixData.GeneralTrackDigiCollectionDM = cms.string("generalTracks")
mixData.GeneralTrackLabelSig = cms.InputTag("generalTracksBeforePreMixing")
mixData.GeneralTrackPileInputTag = cms.InputTag("generalTracksBeforeMixing")
mixData.hitProducer = cms.InputTag("famosSimHits")

# get rid of sistrip and sipixel raw2digi modules run inside DataMixingModule
for p in reversed(range(0,len(mixData.input.producers))):
    module_type =  getattr(mixData.input.producers[p],"@module_type").value()
    for _str in ["SiStrip","SiPixel"]:
        if module_type.find(_str) == 0:
            del mixData.input.producers[p]



