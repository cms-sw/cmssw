import FWCore.ParameterSet.Config as cms

rpcEfficiencyShiftHisto = cms.EDAnalyzer("RPCEfficiencyShiftHisto",
   EffCut = cms.untracked.int32(90),
   GlobalFolder = cms.untracked.string('RPC/RPCEfficiency/'),
   SaveFile = cms.untracked.bool(False),
   NameFile = cms.untracked.string('/afs/cern.ch/user/c/calabria/scratch0/RPCEfficiency.root'),
   NumberOfEndcapDisks = cms.untracked.int32(4)
)
