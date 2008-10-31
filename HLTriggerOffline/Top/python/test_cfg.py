import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")
process.source = cms.Source("PoolSource",
    debugVerbosity = cms.untracked.uint32(10),
    debugFlag = cms.untracked.bool(True),
    fileNames = cms.untracked.vstring(
        '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V5_v1/0001/04D2EB26-1861-DD11-8B21-001731A2845B.root',
        '/store/relval/CMSSW_2_1_0/RelValTTbar/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/IDEAL_V5_v1/0001/B6407084-1561-DD11-B2E7-003048754E4D.root')

)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
)
process.Tracer = cms.Service("Tracer",
    indention = cms.untracked.string('$$')
)

process.myanalysis = cms.EDAnalyzer("TrigAnalyzer",
    TriggerResultsCollection = cms.InputTag("TriggerResults","","HLT")
)

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('test.root')
)

process.p = cms.Path(process.myanalysis)

