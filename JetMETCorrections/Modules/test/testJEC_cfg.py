import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")

process.load('JetMETCorrections.Configuration.DefaultJEC_cff')
process.CondDBCommon.connect = 'sqlite_file:JEC_Summer09_7TeV_ReReco332.db'

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(100)
        )

#process.source = cms.Source("EmptySource")
#############   Define the source file ###############
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('file:///data/kkousour/7EF83E04-22EE-DE11-8FA1-00261894396A.root')
)

process.TFileService=cms.Service("TFileService",fileName=cms.string('histos.root'))
jetPtHistogram = cms.PSet(min          = cms.untracked.double(     10),
                          max          = cms.untracked.double(    200),
                          nbins        = cms.untracked.int32 (     50),
                          name         = cms.untracked.string('JetPt'),
                          description  = cms.untracked.string(     ''),
                          plotquantity = cms.untracked.string(   'pt')
                          )
process.ak5CaloL2L3Histos = cms.EDAnalyzer(
    'CandViewHistoAnalyzer',
    src = cms.InputTag('ak5CaloJetsL2L3'),
    histograms = cms.VPSet(jetPtHistogram)
    )

process.p = cms.Path(process.ak5CaloJetsL2L3 * process.ak5CaloL2L3Histos)










