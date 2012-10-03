import FWCore.ParameterSet.Config as cms

process = cms.Process("Demo")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",

    fileNames = cms.untracked.vstring(
        #'file:/uscms_data/d2/mike1886/ClusterSummary/CMSSW_5_3_3/src/tracks_and_vertices.root'
        'file:/uscms_data/d2/mike1886/ClusterSummary/CMSSW_5_3_3/src/RecoLocalTracker/SubCollectionProducers/myOutputFile.root'
        )
)


#name of the output file containing the tree
process.TFileService = cms.Service("TFileService", fileName = cms.string("summaryTree.root") )

process.MessageLogger.cerr.FwkReport.reportEvery = cms.untracked.int32(1000)

process.demo = cms.EDAnalyzer('ClusterAnalyzer',
                              clusterSum    = cms.InputTag('clusterSummaryProducer')
)


process.p = cms.Path(process.demo)
