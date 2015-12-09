import FWCore.ParameterSet.Config as cms

# dummy implementations of the objects defined by loadDigiAliases and loadTriggerDigiAliases
generalTracks = cms.Sequence()
ecalPreshowerDigis = cms.Sequence()
ecalDigis = cms.Sequence()
hcalDigis = cms.Sequence()
muonDTDigis = cms.Sequence()
muonCSCDigis = cms.Sequence()
muonRPCDigis = cms.Sequence()
gtDigisAliasInfo = cms.Sequence()
gmtDigisAliasInfo = cms.Sequence()

def loadDigiAliases(nopremix = True):

    global generalTracks,ecalPreshowerDigis,ecalDigis,hcalDigis,muonDTDigis,muonCSCDigis,muonRPCDigis

    generalTracks = cms.EDAlias(
        **{"mix" if nopremix else "mixData" 
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
    
    ecalPreShowerDigis = cms.EDAlias(
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
                cms.PSet(type = cms.string("HODataFramesSorted"))
                )
           }
          )

    muonDTDigis = cms.EDAlias(
        **{"simMuonDTDigis" if nopremix else "mixData" :
               cms.VPSet(
                cms.PSet(
                    type = cms.string("DTLayerIdDTDigiMuonDigiCollection")
                    )
                )
           }
          )

    muonRPCDigis = cms.EDAlias(
        **{"simMuonRPCDigis" if nopremix else "mixData" :
               cms.VPSet(
                cms.PSet(
                    type = cms.string("RPCDetIdRPCDigiMuonDigiCollection")
                    )
                )
           }
          )

    muonCSCDigi = cms.EDAlias(
        **{"simMuonCSCDigis" if nopremix else "mixData" :
               cms.VPSet(
                cms.PSet(
                    type = cms.string("CSCDetIdCSCWireDigiMuonDigiCollection"),
                    fromProductInstance = cms.string("MuonCSCWireDigi" if nopremix else "MuonCSCWireDigisDM")),
                cms.PSet(
                    type = cms.string("CSCDetIdCSCStripDigiMuonDigiCollection"),
                    fromProductInstance = cms.string("MuonCSCStripDigi" if nopremix else "MuonCSCStripDigisDM")
                    )
                )
           }
          )

def loadTriggerDigiAliases():

    global gtDigis,gmtDigis

    gtDigis = cms.EDAlias(
        simGtDigis=
        cms.VPSet(
            cms.PSet(type = cms.string("L1GlobalTriggerReadoutRecord")),
            cms.PSet(type = cms.string("L1GlobalTriggerObjectMapRecord"))
            )
        )
    

    gmtDigis = cms.EDAlias (
        simGmtDigisn = 
        cms.VPSet(
            cms.PSet(type = cms.string("L1MuGMTReadoutCollection"))
            )
        )
    

