import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Message Logger
process.MessageLogger = cms.Service("MessageLogger",
    cerr = cms.untracked.PSet(
        enable = cms.untracked.bool(False)
    ),
    cout = cms.untracked.PSet(
        enable = cms.untracked.bool(True)
    )
)

process.load("L1TriggerConfig.L1GeometryProducers.l1CaloGeometry_cfi")
process.load("L1TriggerConfig.L1GeometryProducers.l1CaloGeomRecordSource_cff")

process.load("L1Trigger.GlobalMuonTrigger.gmtDigis_cff")

process.load("L1Trigger.GlobalMuonTrigger.gmttree_cfi")
process.gmttree.GMTInputTag = "gmtDigis"
    
process.source = cms.Source("L1MuGMTHWFileReader",
    fileNames = cms.untracked.vstring("file:gmt_testfile.h4mu.dat")
)

process.gmtDigis.DTCandidates = "source:DT"
process.gmtDigis.CSCCandidates = 'source:CSC'
process.gmtDigis.RPCbCandidates = 'source:RPCb'
process.gmtDigis.RPCfCandidates = 'source:RPCf'
process.gmtDigis.MipIsoData = 'source'
process.gmtDigis.Debug = 0
process.gmtDigis.BX_min = -1
process.gmtDigis.BX_max = 1
process.gmtDigis.BX_min_readout = -1
process.gmtDigis.BX_max_readout = 1

process.p = cms.Path(process.gmtDigis * process.gmttree)
