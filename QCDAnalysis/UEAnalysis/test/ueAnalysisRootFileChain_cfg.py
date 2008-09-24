import FWCore.ParameterSet.Config as cms

process = cms.Process("MBUEAnalysisRootFile")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisParticles_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisTracks_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisJets_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisRootple_cfi")
process.load("QCDAnalysis.UEAnalysis.UERegionSelector_cfi")

process.TFileService = cms.Service("TFileService",
    fileName = cms.string('MBUEAnalysisRootFile.root')
)

process.MessageLogger = cms.Service("MessageLogger",
                                    cerr = cms.untracked.PSet(
    default = cms.untracked.PSet(
    limit = cms.untracked.int32(10)
    )
    ),
                                    cout = cms.untracked.PSet(
    threshold = cms.untracked.string('ERROR')
    # threshold = cms.untracked.string('DEBUG')
    ),
                                    destinations = cms.untracked.vstring('cout')
                                    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
    )
process.source = cms.Source("PoolSource",
#                            fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_1_8/RelValMinBias/GEN-SIM-DIGI-RAW-HLTDEBUG-RECO/STARTUP_V7_v1/0002/221DE0C5-5E82-DD11-A199-000423D98BC4.root')
                            fileNames = cms.untracked.vstring('file:/rdata2/uhh-cms013/data/bechtel/Summer08/CMSSW_2_1_8/src/RelValMinBias-GEN-SIM-DIGI-RAW-HLTDEBUG-RECO-STARTUP_V7_v1-221DE0C5-5E82-DD11-A199-000423D98BC4.root')
                            )

#process.EventAnalyzer = cms.EDAnalyzer("EventContentAnalyzer")

process.UEAnalysisEventContent = cms.OutputModule("PoolOutputModule",
                                                  fileName = cms.untracked.string('UEAnalysisEventContent.root'),
                                                  outputCommands = cms.untracked.vstring('keep *')
                                                  )

#process.p1 = cms.Path(process.UEAnalysisParticles*process.UEAnalysisTracks+process.UEAnalysisJets+process.UEAnalysis)
process.p1 = cms.Path(process.UEAnalysisParticles*process.UEAnalysisTracks+process.UEAnalysisJets+process.UEAnalysis+process.UERegionSelector)
process.UEAnalysisRootple.OnlyRECO = False
process.UEAnalysisRootple500.OnlyRECO = False


