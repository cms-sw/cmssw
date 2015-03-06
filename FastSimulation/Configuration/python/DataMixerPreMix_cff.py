import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.DataMixerPreMix_cff import *

# from signal: mix tracks not strip or pixels
# does this take care of pileup as well?
mixData.TrackerMergeType = "tracks"
import FastSimulation.Tracking.recoTrackAccumulator_cfi
mixData.tracker = FastSimulation.Tracking.recoTrackAccumulator_cfi.recoTrackAccumulator
mixData.hitProducer = cms.InputTag("famosSimHits")

# get rid of sistrip and sipixel raw2digi modules run inside DataMixingModule
for p in reversed(range(0,len(mixData.input.producers))):
    module_type =  getattr(mixData.input.producers[p],"@module_type").value()
    for _str in ["SiStrip","SiPixel"]:
        if module_type.find(_str) == 0:
            del mixData.input.producers[p]



