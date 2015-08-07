import FWCore.ParameterSet.Config as cms

generalTracksAliasInfo = cms.VPSet(
    cms.PSet(
        key = cms.string("mix"),
        value = cms.VPSet( cms.PSet(type=cms.string('recoTracks'),
                                    fromProductInstance = cms.string('generalTracks'),
                                    toProductInstance = cms.string('') ),
                           cms.PSet(type=cms.string('recoTrackExtras'),
                                    fromProductInstance = cms.string('generalTracks'),
                                    toProductInstance = cms.string('') ),
                           cms.PSet(type=cms.string('TrackingRecHitsOwned'),
                                    fromProductInstance = cms.string('generalTracks'),
                                    toProductInstance = cms.string('') ),
                           cms.PSet(type=cms.string('floatedmValueMap'),
                                    fromProductInstance = cms.string('generalTracksMVAVals'),
                                    toProductInstance = cms.string('MVAVals') ) )
        )
    )

ecalPreShowerDigisAliasInfo = cms.VPSet(
    cms.PSet(
        key = cms.string("simEcalPreshowerDigis"),
        value = cms.VPSet(cms.PSet(type = cms.string("ESDigiCollection")))
        )
    )

ecalDigisAliasInfo = cms.VPSet(
    cms.PSet(
        key = cms.string("simEcalDigis"),
        value = cms.VPSet(
            cms.PSet(type = cms.string("EBDigiCollection")),
            cms.PSet(type = cms.string("EEDigiCollection")),
            cms.PSet(
                type = cms.string("EBSrFlagsSorted"),
                fromProductInstance = cms.string('ebSrFlags'),
                toProductInstance = cms.string('')),
            cms.PSet(
                type = cms.string("EESrFlagsSorted"),
                fromProductInstance = cms.string('eeSrFlags'),
                toProductInstance = cms.string(''))),
        ),
    cms.PSet(
        key = cms.string("simEcalTriggerPrimitiveDigis"),
        value = cms.VPSet(
            cms.PSet(
                type = cms.string("EcalTriggerPrimitiveDigisSorted"),
                fromProductInstance = cms.string(""),
                toProductInstance = cms.string("EcalTriggerPrimitives")))
        )
    )

hcalDigisAliasInfo = cms.VPSet(
    cms.PSet(
        key = cms.string("simHcalDigis"),
        value = cms.VPSet(
                cms.PSet(type = cms.string("HBHEDataFramesSorted")),
                cms.PSet(type = cms.string("HFDataFramesSorted")),
                cms.PSet(type = cms.string("HODataFramesSorted")))
        )
    )

muonDTDigisAliasInfo = cms.VPSet(
    cms.PSet(
        key = cms.string("simMuonDTDigis"),
        value = cms.VPSet(cms.PSet(type = cms.string("DTLayerIdDTDigiMuonDigiCollection")))
        )
    )


muonRPCDigisAliasInfo = cms.VPSet(
    cms.PSet(
        key = cms.string("simMuonRPCDigis"),
        value = cms.VPSet(cms.PSet(type = cms.string("RPCDetIdRPCDigiMuonDigiCollection")))
        )
    )

muonCSCDigisAliasInfo = cms.VPSet(
    cms.PSet(
        key = cms.string("simMuonCSCDigis"),
        value = cms.VPSet(
            cms.PSet(
                type = cms.string("CSCDetIdCSCWireDigiMuonDigiCollection"),
                fromProductInstance = cms.string("MuonCSCWireDigi")),
            cms.PSet(
                type = cms.string("CSCDetIdCSCStripDigiMuonDigiCollection"),
                fromProductInstance = cms.string("MuonCSCStripDigi")))
        )
    )

gtDigisAliasInfo = cms.VPSet (
    cms.PSet(
        key = cms.string("simGtDigis"),
        value = cms.VPSet(
            cms.PSet(type = cms.string("L1GlobalTriggerReadoutRecord")),
            cms.PSet(type = cms.string("L1GlobalTriggerObjectMapRecord"))
            )
        )
    )

gmtDigisAliasInfo = cms.VPSet (
    cms.PSet(
        key = cms.string("simGmtDigis"),
        value = cms.VPSet(
            cms.PSet(type = cms.string("L1MuGMTReadoutCollection"))
            )
        )
    )

def convertAliasInfoForDataMixer():
    print "# WARNING: converting digi and mix aliases for DataMixer"
    # tracker
    generalTracksAliasInfo[0].key = "mixData"

    # muon system
    muonCSCDigisAliasInfo[0].key = "mixData"
    muonCSCDigisAliasInfo[0].value[0].fromProductInstance = "MuonCSCWireDigisDM"
    muonCSCDigisAliasInfo[0].value[1].fromProductInstance = "MuonCSCStripDigisDM"
    muonRPCDigisAliasInfo[0].key = "mixData"
    muonDTDigisAliasInfo[0].key = "mixData"

    # calorimeters
    hcalDigisAliasInfo[0].key = "DMHcalDigis"
    ecalDigisAliasInfo[0].key = "DMEcalDigis"
    ecalDigisAliasInfo[1].key = "DMEcalTriggerPrimitiveDigis"
    ecalPreShowerDigisAliasInfo[0].key = "DMEcalPreshowerDigis"

def infoToAlias(info):
    _dict = dict()
    for entry in info:
        _dict[entry.key.value()] = entry.value
    return cms.EDAlias(**_dict)
