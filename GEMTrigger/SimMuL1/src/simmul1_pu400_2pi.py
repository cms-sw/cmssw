import FWCore.ParameterSet.Config as cms

process = cms.Process('simmu')


theNumberOfEvents = 50
theHistoFileName = "h_Zmumu_pu400_2pi.root"


process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(theNumberOfEvents)
)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.load('L1TriggerConfig.L1ScalesProducers.data.L1MuTriggerScalesConfig_cff')
process.load('L1TriggerConfig.L1ScalesProducers.data.L1MuGMTScalesConfig_cff')
process.load('L1TriggerConfig.GMTConfigProducers.data.L1MuGMTParametersConfig_cff')
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load('Configuration.EventContent.EventContent_cff')
process.load("SimGeneral.HepPDTESSource.pythiapdt_cfi")
#process.load('')
#process.load('')

process.options = cms.untracked.PSet(
#     Rethrow = cms.untracked.vstring('ProductNotFound'),
#     FailPath = cms.untracked.vstring('ProductNotFound'),
    makeTriggerResults = cms.untracked.bool(True),
    wantSummary = cms.untracked.bool(True)
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
#    "file:/data1/cmsdata/dimuons/dimuons_wmunu_job1.root",
    "file:/uscmst1b_scratch/lpc1/lpctau/khotilov/slhc/CMSSW_2_2_3/src/out_400.root"
    )
)


process.TFileService = cms.Service("TFileService",
    fileName = cms.string(theHistoFileName)
)


process.SimMuL1StrictAll = cms.EDFilter("SimMuL1",
    doStrictSimHitToTrackMatch = cms.untracked.bool(True),
    matchAllTrigPrimitivesInChamber = cms.untracked.bool(True),

    debugALLEVENT = cms.int32(1),
    debugINHISTOS = cms.int32(1),
    debugALCT = cms.int32(0),
    debugCLCT = cms.int32(0),
    debugLCT = cms.int32(1),
    debugMPLCT = cms.int32(1),
    debugTFTRACK = cms.int32(1),
    debugTFCAND = cms.int32(1),
    debugGMTCAND = cms.int32(1),
    debugL1EXTRA = cms.int32(1),
    debugRATE = cms.int32(1),
    
    minSimTrPt = cms.untracked.double(2.),
    minSimTrPhi = cms.untracked.double(-9.),
    maxSimTrPhi = cms.untracked.double(9.),
    minSimTrEta = cms.untracked.double(0.8),
    maxSimTrEta = cms.untracked.double(2.5),
    invertSimTrPhiEta = cms.untracked.bool(False),
    
    GMTPS = cms.PSet(
        DTCandidates = cms.InputTag("simDttfDigis:DT")
        CSCCandidates = cms.InputTag("simCsctfDigis:CSC")
        RPCbCandidates = cms.InputTag("simRpcTriggerDigis:RPCb")
        RPCfCandidates = cms.InputTag("simRpcTriggerDigis:RPCf")
        MipIsoData = L1RCTRegionSumsEmCands
        Debug = cms.int32(),
        BX_min = cms.int32(-4),
        BX_max = cms.int32(4),
        BX_min_readout = cms.int32(-2),
        BX_max_readout = cms.int32(2),
	WriteLUTsAndRegs = cms.untracked.bool(True)
    )
)

#process.output = cms.OutputModule("PoolOutputModule",
#)

# Other statements
process.GlobalTag.globaltag = 'IDEAL_V11::All'

#process.Timing = cms.Service("Timing")
process.Tracer = cms.Service("Tracer")


process.ana_seq = cms.Sequence(process.SimMuL1StrictAll)

process.ana_step        = cms.EndPath(process.ana_seq)

process.schedule = cms.Schedule(
    process.ana_step
)
