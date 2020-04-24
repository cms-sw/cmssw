import FWCore.ParameterSet.Config as cms

#InputFile = "file:/afs/cern.ch/user/z/zhangj/private/CaloLayer2Emulatior/CMSSW_7_6_0_pre4/src/L1Trigger/L1TCalorimeter/test/l1tCalo_2016_simEDM.root"
InputFile = "file:/afs/cern.ch/user/z/zhangj/private/DQM/CMSSW_7_6_0_pre7/src/L1Trigger/L1TCalorimeter/test/l1tCalo_2016_simEDM_260627.root"

OutputFile="comparared.root"
nevts=-1

process = cms.Process("DQM")

process.load("Configuration.StandardSequences.RawToDigi_cff")

process.load("L1Trigger.HardwareValidation.L1ComparatorRun2_cfi")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(InputFile )
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32( nevts )
)

process.out = cms.OutputModule("PoolOutputModule",
    outputCommands = cms.untracked.vstring(
        'keep *',
    ),
    fileName = cms.untracked.string(OutputFile)
)

process.o = cms.EndPath( process.out )

#process.unpacker = cms.Path( process.caloStage1Digis +  process.l1GctHwDigis)
process.comparator = cms.Path( process.caloStage2Digis + process.l1comparatorResultDigis)
process.schedule = cms.Schedule(process.comparator,
                                process.o)
