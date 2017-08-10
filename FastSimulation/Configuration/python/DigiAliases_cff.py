import FWCore.ParameterSet.Config as cms

# define some global variables
# to be filled in by the load* functions below
generalTracks = None
ecalPreshowerDigis = None
ecalDigis = None
hcalDigis = None
muonDTDigis = None
muonCSCDigis = None
muonRPCDigis = None
caloStage1LegacyFormatDigis = None
gtDigis = None
gmtDigis = None

def loadDigiAliases(premix=False):

    nopremix = not premix

    global generalTracks,ecalPreshowerDigis,ecalDigis,hcalDigis,muonDTDigis,muonCSCDigis,muonRPCDigis

    generalTracks = cms.EDAlias(
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
    
    ecalPreshowerDigis = cms.EDAlias(
        **{"simEcalPreshowerDigis" if nopremix else "DMEcalPreshowerDigis" :
               cms.VPSet(
                cms.PSet(
                    type = cms.string("ESDigiCollection")
                    )
                )
           }
          )
    
    ecalDigis = cms.EDAlias(
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

    hcalDigis = cms.EDAlias(
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

    muonDTDigis = cms.EDAlias(
        **{"simMuonDTDigis" if nopremix else "mixData" :
               cms.VPSet(
                cms.PSet(
                    type = cms.string("DTLayerIdDTDigiMuonDigiCollection")
                    ),
                #cms.PSet(
                #    type = cms.string("DTLayerIdDTDigiSimLinkMuonDigiCollection")
                #    )
                )
           }
          )

    muonRPCDigis = cms.EDAlias(
        **{"simMuonRPCDigis" if nopremix else "mixData" :
               cms.VPSet(
                cms.PSet(
                    type = cms.string("RPCDetIdRPCDigiMuonDigiCollection")
                    ),
                #cms.PSet(
                #    type = cms.string("RPCDigiSimLinkedmDetSetVector")
                #    )
                )
           }
          )

    muonCSCDigis = cms.EDAlias(
        **{"simMuonCSCDigis" if nopremix else "mixData" :
               cms.VPSet(
                cms.PSet(
                    type = cms.string("CSCDetIdCSCWireDigiMuonDigiCollection"),
                    fromProductInstance = cms.string("MuonCSCWireDigi" if nopremix else "MuonCSCWireDigisDM"),
                    toProductInstance = cms.string("MuonCSCWireDigi")),
                cms.PSet(
                    type = cms.string("CSCDetIdCSCStripDigiMuonDigiCollection"),
                    fromProductInstance = cms.string("MuonCSCStripDigi" if nopremix else "MuonCSCStripDigisDM"),
                    toProductInstance = cms.string("MuonCSCStripDigi")),
                #cms.PSet(
                #    type = cms.string('StripDigiSimLinkedmDetSetVector')
                #    ),
                )
           }
          )
    
def loadTriggerDigiAliases():

    global gctDigis,gtDigis,gmtDigis,caloStage1LegacyFormatDigis

    caloStage1LegacyFormatDigis = cms.EDAlias(
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

    gctDigis = cms.EDAlias(
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

    gtDigis = cms.EDAlias(
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
    

    gmtDigis = cms.EDAlias (
        simGmtDigis = 
        cms.VPSet(
            cms.PSet(type = cms.string("L1MuGMTReadoutCollection")),
            cms.PSet(type = cms.string("L1MuGMTCands"))
            )
        )
    
