import FWCore.ParameterSet.Config as cms

process = cms.Process("L1MicroGMTEmulator")

process.load("FWCore.MessageService.MessageLogger_cfi")


process.source = cms.Source("EmptySource")
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1))

process.load('L1Trigger.L1TMuon.fakeGmtParams_cff')
#process.gmtParams.BrlSingleMatchQualLUTPath = cms.string('')
#process.gmtParams.FwdPosSingleMatchQualLUTPath = cms.string('')
#process.gmtParams.FwdNegSingleMatchQualLUTPath = cms.string('')
#process.gmtParams.OvlPosSingleMatchQualLUTPath = cms.string('')
#process.gmtParams.OvlNegSingleMatchQualLUTPath = cms.string('')
#process.gmtParams.BOPosMatchQualLUTPath = cms.string('')
#process.gmtParams.BONegMatchQualLUTPath = cms.string('')
#process.gmtParams.FOPosMatchQualLUTPath = cms.string('')
#process.gmtParams.FONegMatchQualLUTPath = cms.string('')
#process.gmtParams.SortRankLUTPath = cms.string('')

process.dumper = cms.EDAnalyzer("L1TMicroGMTLUTDumper",
    out_directory = cms.string("lut_dump"),
)

process.dumpPath = cms.Path( process.dumper )
process.schedule = cms.Schedule(process.dumpPath)
