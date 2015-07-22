import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.DataMixerPreMix_cff import *

# from signal: mix tracks not strip or pixels
# does this take care of pileup as well?
mixData.TrackerMergeType = "tracks"
import FastSimulation.Tracking.recoTrackAccumulator_cfi
mixData.tracker = FastSimulation.Tracking.recoTrackAccumulator_cfi.recoTrackAccumulator.clone()
mixData.tracker.pileUpTracks = cms.InputTag("mix","generalTracks")
mixData.hitsProducer = "famosSimHits"

# give digi collections the names expected by RECO and HLT
import FastSimulation.Configuration.DigiAndMixAliasInfo_cff as _aliasInfo
_aliasInfo.convertAliasInfoForDataMixer()
generalTracks = _aliasInfo.infoToAlias(_aliasInfo.generalTracksAliasInfo)
ecalPreshowerDigis = _aliasInfo.infoToAlias(_aliasInfo.ecalPreShowerDigisAliasInfo)
ecalDigis = _aliasInfo.infoToAlias(_aliasInfo.ecalDigisAliasInfo)
hcalDigis = _aliasInfo.infoToAlias(_aliasInfo.hcalDigisAliasInfo)
muonDTDigis = _aliasInfo.infoToAlias(_aliasInfo.muonDTDigisAliasInfo)
muonCSCDigis = _aliasInfo.infoToAlias(_aliasInfo.muonCSCDigisAliasInfo)
muonRPCDigis = _aliasInfo.infoToAlias(_aliasInfo.muonRPCDigisAliasInfo)

