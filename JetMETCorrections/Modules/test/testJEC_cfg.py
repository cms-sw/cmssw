import FWCore.ParameterSet.Config as cms

process = cms.Process("myprocess")


process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START38_V6::All'
process.GlobalTag.connect = 'sqlite_file:START38_V6.db'
process.load('JetMETCorrections.Configuration.DefaultJEC_cff')

process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(100)
        )

#process.source = cms.Source("EmptySource")
#############   Define the source file ###############
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring('/store/relval/CMSSW_3_8_0_pre7/RelValTTbar/GEN-SIM-RECO/START38_V4-v1/0002/DC19EA07-4286-DF11-BD2B-0030487CD16E.root')
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










