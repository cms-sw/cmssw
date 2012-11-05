import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        'file:reco2012C.root'
#        'file:reco2012_New.root'
    )
)

# process.TFileService = cms.Service("TFileService",fileName=cms.string('vHisto.root'))


process.demo = cms.EDAnalyzer('TestEgammaTowerIsolation')
process.mypath = cms.Path(process.demo)
process.schedule = cms.Schedule(process.mypath)

