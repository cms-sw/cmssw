import FWCore.ParameterSet.Config as cms

process = cms.Process("MBUEAnalysisRootFile")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisParticles_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisTracks_cfi")
process.load("QCDAnalysis.UEAnalysis.UEAnalysisJets_cfi")
process.load("QCDAnalysis.UEAnalysis.UERegionSelector_cfi")

#process.MessageLogger = cms.Service("MessageLogger",
#                                    cerr = cms.untracked.PSet(
#    default = cms.untracked.PSet(
#    limit = cms.untracked.int32(10)
#    )
#    ),
#                                    cout = cms.untracked.PSet(
#    threshold = cms.untracked.string('ERROR')
#    # threshold = cms.untracked.string('DEBUG')
#    ),
#                                    destinations = cms.untracked.vstring('cout')
#                                    )

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(100)
    )
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring(
#    'file:/rdata2/uhh-cms013/data/bechtel/Summer08/CMSSW_2_1_9/src/Summer08-MinBias-02328B76-C88B-DD11-8DD5-003048770348.root'
    'file:/rdata2/uhh-cms013/data/bechtel/Summer08/CMSSW_2_1_9/src/Summer08-MinBias900GeV-00398733-DA89-DD11-9005-001EC9AA91F8.root'
##    'file:/rdata2/uhh-cms013/data/bechtel/Summer08/CMSSW_2_1_9/src/RelValQCD_Pt_80_120-Ideal-000AD2A4-6E86-DD11-AA99-000423D9863C.root'
#    '/store/mc/Summer08/HerwigQCDPt15/GEN-SIM-RECO/IDEAL_V9_v1/0000/04631626-4493-DD11-A07F-00D0680BF8C7.root'
#'/store/mc/Summer08/HerwigQCDPt30/GEN-SIM-RECO/IDEAL_V9_v1/0000/00671731-5B93-DD11-9F71-00D0680BF97E.root'
#'/store/mc/Summer08/HerwigQCDPt80/GEN-SIM-RECO/IDEAL_V9_v1/0005/007032F8-439C-DD11-B1B4-0030489847AB.root'
#'/store/mc/Summer08/HerwigQCDPt170/GEN-SIM-RECO/IDEAL_V9_v1/0005/661F442D-30A4-DD11-B881-00145E55647F.root'
#'/store/mc/Summer08/HerwigQCDPt300/GEN-SIM-RECO/IDEAL_V9_v1/0006/04189352-BFA5-DD11-A8A5-00D0680BF8C3.root'
#'/store/mc/Summer08/HerwigQCDPt470/GEN-SIM-RECO/IDEAL_V9_v1/0005/08F405A1-7DA0-DD11-91A7-001560EDC951.root'

    )
                            )

#process.EventAnalyzer = cms.EDAnalyzer("EventContentAnalyzer")

process.UEAnalysisEventContent = cms.OutputModule("PoolOutputModule",
                                                  fileName = cms.untracked.string('UEAnalysisEventContent.root'),
                                                  outputCommands = cms.untracked.vstring('keep *')
                                                  )

process.p1 = cms.Path(process.UEAnalysisParticles*process.UEAnalysisTracks+process.UEAnalysisJets+process.UERegionSelector)
process.UEPoolOutput = cms.EndPath(process.UEAnalysisEventContent)
