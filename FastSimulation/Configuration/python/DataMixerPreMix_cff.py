import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.DataMixerPreMix_cff import *

# from signal: mix tracks not strip or pixels
# does this take care of pileup as well?
mixData.TrackerMergeType = "tracks"
import FastSimulation.Tracking.recoTrackAccumulator_cfi
mixData.tracker = FastSimulation.Tracking.recoTrackAccumulator_cfi.recoTrackAccumulator.clone()
mixData.tracker.pileUpTracks = cms.InputTag("mix","generalTracks")
mixData.tracker.pileUpMVAValues = cms.InputTag("mix","generalTracksMVAVals")
mixData.hitProducer = cms.InputTag("famosSimHits")

# get rid of sistrip and sipixel raw2digi modules run inside DataMixingModule
for p in reversed(range(0,len(mixData.input.producers))):
    module_type =  getattr(mixData.input.producers[p],"@module_type").value()
    for _str in ["SiStrip","SiPixel"]:
        if module_type.find(_str) == 0:
            del mixData.input.producers[p]

# extend it with some digi2raw
from Configuration.StandardSequences.DigiToRawDM_cff import ecalPacker,esDigiToRaw,hcalRawData,rawDataCollector
for _entry in [cms.InputTag("SiStripDigiToRaw"), cms.InputTag("castorRawData"),cms.InputTag("siPixelRawData")]:
    rawDataCollector.RawCollectionList.remove(_entry)

# extend it with some raw2digi
from Configuration.StandardSequences.RawToDigi_cff import ecalPreshowerDigis,ecalDigis,hcalDigis


# aliases for muons
muonDTDigis = cms.EDAlias(
    mixData = cms.VPSet(
        cms.PSet(type = cms.string("DTLayerIdDTDigiMuonDigiCollection"))
        )
    )
muonRPCDigis = cms.EDAlias(
    mixData = cms.VPSet(
        cms.PSet(type = cms.string("RPCDetIdRPCDigiMuonDigiCollection"))
        )
    )

muonCSCDigis = cms.EDAlias(
    mixData = cms.VPSet(
        cms.PSet(
            type = cms.string("CSCDetIdCSCWireDigiMuonDigiCollection"),
            fromProductInstance = cms.string("MuonCSCWireDigisDM"),
            ),
        cms.PSet(
            type = cms.string("CSCDetIdCSCStripDigiMuonDigiCollection"),
            fromProductInstance = cms.string("MuonCSCStripDigisDM"),
            )
        )
    )

from FastSimulation.Tracking.GeneralTracksAlias_cfi import generalTracksAliasInfo
generalTracksAliasInfo.key = "mixData"
generalTracks = cms.EDAlias(**{generalTracksAliasInfo.key.value():generalTracksAliasInfo.value})

calDigiToRawToDigi = cms.Sequence(ecalPacker+esDigiToRaw+hcalRawData+rawDataCollector+ecalPreshowerDigis+ecalDigis+hcalDigis)
pdatamix.replace(postDMDigi,postDMDigi+calDigiToRawToDigi)
