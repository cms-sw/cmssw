import FWCore.ParameterSet.Config as cms

process = cms.Process("MBUEAnalysisRootFile")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisParticles_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisTracks_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisJets_cfi")
process.load("QCDAnalysis.UEAnalysis.UERegionSelector_cfi")

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
    input = cms.untracked.int32(10)
    )
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:/rdata2/uhh-cms013/data/bechtel/Summer08/CMSSW_2_1_9/src/RelValQCD_Pt_80_120-Ideal-000AD2A4-6E86-DD11-AA99-000423D9863C.root')
                            )

#process.EventAnalyzer = cms.EDAnalyzer("EventContentAnalyzer")

process.UEAnalysisEventContent = cms.OutputModule("PoolOutputModule",
                                                  fileName = cms.untracked.string('UEAnalysisEventContent.root'),
#                                                  outputCommands  = cms.untracked.vstring('keep *_towardsTracks_*_*','keep *_transverseTracks_*_*','keep *_awayTracks_*_*','drop *_LeadingIC5TracksJet_*_*')
                                                  outputCommands = cms.untracked.vstring('keep *','drop *_LeadingIC5TracksJet_*_*')
#                                                  outputCommands = cms.untracked.vstring('keep *')
                                                  )

process.p1 = cms.Path(process.UEAnalysisParticles*process.UEAnalysisTracks+process.UEAnalysisJets+process.UERegionSelector)
process.UEPoolOutput = cms.EndPath(process.UEAnalysisEventContent)
