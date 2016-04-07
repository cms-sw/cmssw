import FWCore.ParameterSet.Config as cms

OutALCARECORpcCalHLT_noDrop = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECORpcCalHLT')
    ),
    outputCommands = cms.untracked.vstring(
        'keep *_muonDTDigis_*_*', 
        #'keep CSCDetIdCSCWireDigiMuonDigiCollection_*_*_*', 
        #'keep CSCDetIdCSCStripDigiMuonDigiCollection_*_*_*', 
        'keep CSCDetIdCSCWireDigiMuonDigiCollection_muonCSCDigis_*_*', 
        'keep CSCDetIdCSCStripDigiMuonDigiCollection_muonCSCDigis_*_*', 
        #'keep DTLayerIdDTDigiMuonDigiCollection_*_*_*', 
        'keep DTLayerIdDTDigiMuonDigiCollection_muonDTDigis_*_*', 
        'keep *_dt4DSegments_*_*',
        'keep *_cscSegments_*_*',
        'keep *_rpcRecHits_*_*',        
        #'keep RPCDetIdRPCDigiMuonDigiCollection_*_*_*', 
        'keep RPCDetIdRPCDigiMuonDigiCollection_muonRPCDigis_*_*', 
        'keep recoMuons_muonsNoRPC_*_*',                               
        'keep L1MuRegionalCands_*_RPCb_*',
        'keep L1MuRegionalCands_*_RPCf_*',
        'keep L1MuGMTCands_*_*_*', 
        'keep L1MuGMTReadoutCollection_*_*_*')
)



import copy
OutALCARECORpcCalHLT=copy.deepcopy(OutALCARECORpcCalHLT_noDrop)
OutALCARECORpcCalHLT.outputCommands.insert(0, "drop *")
