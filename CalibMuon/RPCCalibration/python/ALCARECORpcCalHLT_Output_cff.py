import FWCore.ParameterSet.Config as cms

OutALCARECORpcCalHLT = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('pathALCARECORpcCalHLT')
    ),
    outputCommands = cms.untracked.vstring('drop *', 
        'keep *_muonDTDigis_*_*', 
        'keep CSCDetIdCSCWireDigiMuonDigiCollection_*_*_*', 
        'keep CSCDetIdCSCStripDigiMuonDigiCollection_*_*_*', 
        'keep DTLayerIdDTDigiMuonDigiCollection_*_*_*', 
        'keep *_dt4DSegments_*_*',
        'keep *_cscSegments_*_*',
        'keep *_rpcRecHits_*_*',        
        'keep RPCDetIdRPCDigiMuonDigiCollection_*_*_*', 
        'keep recoMuons_muonsNoRPC_*_*',                               
        'keep L1MuRegionalCands_*_RPCb_*',
        'keep L1MuRegionalCands_*_RPCf_*',
        'keep L1MuGMTCands_*_*_*', 
        'keep L1MuGMTReadoutCollection_*_*_*')
)

