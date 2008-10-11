import FWCore.ParameterSet.Config as cms

rpcAfterPulse = cms.EDAnalyzer("RPCMon_SS_Dbx_Global",
   GlobalHistogramsFolder = cms.untracked.string('RPC/RecHits/SummaryHistogram'),
   RootFileName =  cms.untracked.string('out.root'),
   SaveRootFile = cms.untracked.bool(False)
)
