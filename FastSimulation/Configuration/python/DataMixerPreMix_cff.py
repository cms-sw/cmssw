import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.DataMixerPreMix_cff import *

# from signal: mix tracks not strip or pixels
# does this take care of pileup as well?
mixData.TrackerMergeType = "tracks"
import FastSimulation.Tracking.recoTrackAccumulator_cfi
mixData.tracker = FastSimulation.Tracking.recoTrackAccumulator_cfi.recoTrackAccumulator.clone()
mixData.tracker.pileUpTracks = cms.InputTag("mix","generalTracks")
mixData.tracker.pileUpMVAValues = cms.InputTag("mix","generalTracksMVAVals")
mixData.hitsProducer = "famosSimHits"

# give digi collections the names expected by RECO and HLT
ecalPreshowerDigis = cms.EDAlias(
    DMEcalPreshowerDigis = cms.VPSet(
        cms.PSet(type = cms.string("ESDigiCollection"))
        )
    )

ecalDigis = cms.EDAlias(
    DMEcalDigis = cms.VPSet(
        cms.PSet(type = cms.string("EBDigiCollection")),
        cms.PSet(type = cms.string("EEDigiCollection"))
        )
    )

hcalDigis = cms.EDAlias(
    DMHcalDigis = cms.VPSet(
        cms.PSet(type = cms.string("HBHEDataFramesSorted")),
        cms.PSet(type = cms.string("HFDataFramesSorted")),
        cms.PSet(type = cms.string("HODataFramesSorted"))
        )
    )

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
