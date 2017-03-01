import FWCore.ParameterSet.Config as cms

l1Muon_EventContent = cms.PSet(
    outputCommands = cms.untracked.vstring('drop *', 
        'keep CSCDetIdCSCWireDigiMuonDigiCollection_*_*_*', 
        'keep CSCDetIdCSCStripDigiMuonDigiCollection_*_*_*', 
        'keep DTLayerIdDTDigiMuonDigiCollection_*_*_*', 
        #'keep RPCDetIdRPCDigiMuonDigiCollection_*_*_*', 
        'keep RPCDetIdRPCDigiMuonDigiCollection_muonRPCDigis_*_*', 
        'keep RPCDetIdRPCDigiMuonDigiCollection_simMuonRPCDigis_*_*', 
        'keep L1MuGMTCands_*_*_*', 
        'keep L1MuGMTReadoutCollection_*_*_*')
)
l1MuonEventSelection = cms.PSet(
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('l1MuonHLTPath')
    )
)

