import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(10) )

process.DQMStore = cms.Service("DQMStore")


process.source = cms.Source("PoolSource",
    # replace 'myfile.root' with the source file you want to use
    fileNames = cms.untracked.vstring(

'file:/afs/cern.ch/user/l/lantonel/scratch0/CMSSW_3_0_0_pre6/src/DQMOffline/EGamma/promptrecoCosmics.root'
    )
)

process.demo = cms.EDAnalyzer('PhotonDataCertification')

process.p = cms.Path(process.demo)
