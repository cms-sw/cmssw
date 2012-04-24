import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",

    fileNames = cms.untracked.vstring(
        'file:myOutputFile.root'
        )
)


#name of the output file containing the tree
process.TFileService = cms.Service("TFileService", fileName = cms.string("summaryTree.root") )

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)

process.demo = cms.EDAnalyzer('ClusterAnalyzer',
                              clusterSum    = cms.InputTag('clusterSummaryProducer')
)


process.p = cms.Path(process.demo)
