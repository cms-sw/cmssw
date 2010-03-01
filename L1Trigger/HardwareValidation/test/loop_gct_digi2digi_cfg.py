# Description:
#  test gct sequence: digi -> raw -> digi
#  check consistency of input and output rct,gct digi collections

import FWCore.ParameterSet.Config as cms

process = cms.Process("gctDigiToDigi")

process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.load("L1Trigger.HardwareValidation.L1Comparator_cfi")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:calodigis.root')
)

process.gctDigiToRaw = cms.EDProducer("GctDigiToRaw",
    rctInputLabel = cms.InputTag("rctDigis"),
    gctInputLabel = cms.InputTag("gctDigis"),
    gctFedId = cms.int32(745),
    verbose = cms.untracked.bool(False)
)

process.l1GctHwDigis = cms.EDProducer("GctRawToDigi",
    gctFedId = cms.int32(745),
    unpackInternEm = cms.untracked.bool(True),
    inputLabel = cms.InputTag("gctDigiToRaw"),
    verbose = cms.untracked.bool(False),
    unpackFibres = cms.untracked.bool(True)
)

process.dump = cms.EDAnalyzer("DumpFEDRawDataProduct",
    feds = cms.untracked.vint32(745),
    dumpPayload = cms.untracked.bool(True)
)

process.l1compare.GCTsourceEmul = 'gctDigis'                          
process.l1compare.GCTsourceData = 'l1GctHwDigis'                      
process.l1compare.DumpFile = 'dump.txt'                               
process.l1compare.DumpMode = 1                                        
                                                                      
process.l1compare.COMPARE_COLLS = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]

process.outputEvents = cms.OutputModule("PoolOutputModule",
    fileName = cms.untracked.string('gctDigiToDigi.root')
)

process.p = cms.Path(
    process.gctDigiToRaw
    *process.dump
    *process.l1GctHwDigis
    *process.l1compare
    )

process.outpath = cms.EndPath(process.outputEvents)                   
                                                                      


