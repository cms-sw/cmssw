import FWCore.ParameterSet.Config as cms

rpcAfterPulse = cms.EDAnalyzer("RPCMon_SS_Dbx_Global",
   GlobalHistogramsFolder = cms.untracked.string('RPC/RecHits/SummaryHistograms'),
   RootFileName =  cms.untracked.string('out.root'),
   SaveRootFile = cms.untracked.bool(False),
   rpcDigiCollectionTag = cms.InputTag('rpcunpacker')
)
