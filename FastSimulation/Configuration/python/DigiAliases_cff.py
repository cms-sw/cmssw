import FWCore.ParameterSet.Config as cms

# This is an ugly hack (but better what was before) to record if the
# loadDigiAliases() was called with premixing or not. Unfortunately
# which alias to use depends on that. If we had a premixing Modifier,
# this hack would not be needed.
_loadDigiAliasesWasCalledPremix = None

def loadGeneralTracksAlias(process):
    if _loadDigiAliasesWasCalledPremix is None:
        raise Exception("This function may be called only after loadDigiAliases() has been called")

    nopremix = not _loadDigiAliasesWasCalledPremix
    process.generalTracks = cms.EDAlias(
        **{"mix" if nopremix else "mixData" :
           cms.VPSet(
                cms.PSet(
                    fromProductInstance = cms.string('generalTracks'),
                    toProductInstance = cms.string(''),
                    type = cms.string('recoTracks')
                    ), 
                cms.PSet(
                    fromProductInstance = cms.string('generalTracks'),
                    toProductInstance = cms.string(''),
                    type = cms.string('recoTrackExtras')
                    ), 
                cms.PSet(
                    fromProductInstance = cms.string('generalTracks'),
                    toProductInstance = cms.string(''),
                    type = cms.string('TrackingRecHitsOwned')
                    )
                )
           }
          )

def loadDigiAliases(process, premix=False):
    nopremix = not premix
    global _loadDigiAliasesWasCalledPremix
    _loadDigiAliasesWasCalledPremix = premix

    process.ecalPreshowerDigis = cms.EDAlias(
        **{"simEcalPreshowerDigis" if nopremix else "DMEcalPreshowerDigis" :
               cms.VPSet(
                cms.PSet(
                    type = cms.string("ESDigiCollection")
                    )
                )
           }
          )
    
    process.ecalDigis = cms.EDAlias(
        **{"simEcalDigis" if nopremix else "DMEcalDigis" : 
           cms.VPSet(
                cms.PSet(
                    type = cms.string("EBDigiCollection")
                    ),
                cms.PSet(
                    type = cms.string("EEDigiCollection")
                    ),
                cms.PSet(
                    type = cms.string("EBSrFlagsSorted"),
                    fromProductInstance = cms.string('ebSrFlags'),
                    toProductInstance = cms.string('')),
                cms.PSet(
                    type = cms.string("EESrFlagsSorted"),
                    fromProductInstance = cms.string('eeSrFlags'),
                    toProductInstance = cms.string(''),
                    )
                ),
           "simEcalTriggerPrimitiveDigis" if nopremix else "DMEcalTriggerPrimitiveDigis" :
           cms.VPSet(
                cms.PSet(
                    type = cms.string("EcalTriggerPrimitiveDigisSorted"),
                    fromProductInstance = cms.string(""),
                    toProductInstance = cms.string("EcalTriggerPrimitives")
                    )
                )
           }
          )

    process.hcalDigis = cms.EDAlias(
        **{"simHcalDigis" if nopremix else "DMHcalDigis" :
            cms.VPSet(
                cms.PSet(type = cms.string("HBHEDataFramesSorted")),
                cms.PSet(type = cms.string("HFDataFramesSorted")),
                cms.PSet(type = cms.string("HODataFramesSorted")),
                cms.PSet(
                    type = cms.string('QIE10DataFrameHcalDataFrameContainer'),
                    fromProductInstance = cms.string('HFQIE10DigiCollection'),
                    toProductInstance = cms.string('')
                ),
                cms.PSet(
                    type = cms.string('QIE11DataFrameHcalDataFrameContainer'),
                    fromProductInstance = cms.string('HBHEQIE11DigiCollection'),
                    toProductInstance = cms.string('')
                )
            )
           }
          )

    process.muonDTDigis = cms.EDAlias(
        simMuonDTDigis = cms.VPSet(
                cms.PSet(
                    type = cms.string("DTLayerIdDTDigiMuonDigiCollection")
                    ),
                #cms.PSet(
                #    type = cms.string("DTLayerIdDTDigiSimLinkMuonDigiCollection")
                #    )
                )
          )

    process.muonRPCDigis = cms.EDAlias(
        simMuonRPCDigis = cms.VPSet(
                cms.PSet(
                    type = cms.string("RPCDetIdRPCDigiMuonDigiCollection")
                    ),
                #cms.PSet(
                #    type = cms.string("RPCDigiSimLinkedmDetSetVector")
                #    )
                )
          )

    process.muonCSCDigis = cms.EDAlias(
        simMuonCSCDigis = cms.VPSet(
                cms.PSet(
                    type = cms.string("CSCDetIdCSCWireDigiMuonDigiCollection"),
                    fromProductInstance = cms.string("MuonCSCWireDigi"),
                    toProductInstance = cms.string("MuonCSCWireDigi")),
                cms.PSet(
                    type = cms.string("CSCDetIdCSCStripDigiMuonDigiCollection"),
                    fromProductInstance = cms.string("MuonCSCStripDigi"),
                    toProductInstance = cms.string("MuonCSCStripDigi")),
                #cms.PSet(
                #    type = cms.string('StripDigiSimLinkedmDetSetVector')
                #    ),
                )
          )
    
def loadTriggerDigiAliases(process):
    process.caloStage1LegacyFormatDigis = cms.EDAlias(
        **{ "simCaloStage1LegacyFormatDigis" :
                cms.VPSet(
                cms.PSet(type = cms.string("L1GctEmCands")),
                cms.PSet(type = cms.string("L1GctEtHads")),
                cms.PSet(type = cms.string("L1GctEtMisss")),
                cms.PSet(type = cms.string("L1GctEtTotals")),
                cms.PSet(type = cms.string("L1GctHFBitCountss")),
                cms.PSet(type = cms.string("L1GctHFRingEtSumss")),
                cms.PSet(type = cms.string("L1GctHtMisss")),
                cms.PSet(type = cms.string("L1GctInternEtSums")),
                cms.PSet(type = cms.string("L1GctInternHtMisss")),
                cms.PSet(type = cms.string("L1GctInternJetDatas")),
                cms.PSet(type = cms.string("L1GctJetCands")))})

    process.gctDigis = cms.EDAlias(
        **{ "simGctDigis" :
                cms.VPSet(
                cms.PSet(type = cms.string("L1GctEmCands")),
                cms.PSet(type = cms.string("L1GctEtHads")),
                cms.PSet(type = cms.string("L1GctEtMisss")),
                cms.PSet(type = cms.string("L1GctEtTotals")),
                cms.PSet(type = cms.string("L1GctHFBitCountss")),
                cms.PSet(type = cms.string("L1GctHFRingEtSumss")),
                cms.PSet(type = cms.string("L1GctHtMisss")),
                cms.PSet(type = cms.string("L1GctInternEtSums")),
                cms.PSet(type = cms.string("L1GctInternHtMisss")),
                cms.PSet(type = cms.string("L1GctInternJetDatas")),
                cms.PSet(type = cms.string("L1GctJetCands")))})

    process.gtDigis = cms.EDAlias(
        **{ "simGtDigis" :
                cms.VPSet(
                cms.PSet(type = cms.string("L1GlobalTriggerEvmReadoutRecord")),
                cms.PSet(type = cms.string("L1GlobalTriggerObjectMapRecord")),
                cms.PSet(type = cms.string("L1GlobalTriggerReadoutRecord"))),
            "simGmtDigis" :
                cms.VPSet(
                cms.PSet(type = cms.string("L1MuGMTReadoutCollection")),
                cms.PSet(type = cms.string("L1MuGMTCands")))
            })
    

    process.gmtDigis = cms.EDAlias (
        simGmtDigis = 
        cms.VPSet(
            cms.PSet(type = cms.string("L1MuGMTReadoutCollection")),
            cms.PSet(type = cms.string("L1MuGMTCands"))
            )
        )
    
