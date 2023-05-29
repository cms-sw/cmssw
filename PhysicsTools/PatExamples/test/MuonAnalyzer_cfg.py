import FWCore.ParameterSet.Config as cms

process = cms.Process("RecoMuon")
# Messages
process.load("FWCore.MessageService.MessageLogger_cfi")


process.load("Configuration.StandardSequences.GeometryDB_cff")
process.load("Configuration.StandardSequences.MagneticField_38T_cff")
process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")

process.GlobalTag.globaltag = cms.string( autoCond[ 'phase1_2022_realistic' ] )
process.source = cms.Source("PoolSource",
                            fileNames = cms.untracked.vstring('file:PATLayer1_Output.fromAOD_full.root')
)

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

process.TFileService=cms.Service('TFileService',
                                 fileName=cms.string('MyMuonPlots.root')
                                 )


process.muonAnalysis = cms.EDAnalyzer("ExampleMuonAnalyzer",
                                      MuonCollection = cms.untracked.InputTag('cleanLayer1Muons'),
                                      )


process.p = cms.Path(process.muonAnalysis)




