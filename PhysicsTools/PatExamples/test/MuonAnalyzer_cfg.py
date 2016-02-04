import FWCore.ParameterSet.Config as cms

process = cms.Process("RecoMuon")
# Messages
process.load("FWCore.MessageService.MessageLogger_cfi")


process.load("Configuration.StandardSequences.Geometry_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.GlobalTag.globaltag = 'IDEAL_V9::All'

process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring('/store/relval/CMSSW_2_2_1/RelValH200ZZ4L/GEN-SIM-RECO/IDEAL_V9_v1/0004/B492FC1B-06C5-DD11-93B9-000423D33970.root')
                            fileNames = cms.untracked.vstring('file:PATLayer1_Output.fromAOD_full.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.TFileService=cms.Service('TFileService',
                                 fileName=cms.string('MyMuonPlots.root')
                                 )


process.muonAnalysis = cms.EDAnalyzer("ExampleMuonAnalyzer",
                                      MuonCollection = cms.untracked.string('cleanLayer1Muons'),
                                      )


process.p = cms.Path(process.muonAnalysis)




