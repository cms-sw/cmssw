import FWCore.ParameterSet.Config as cms
source = cms.Source("PoolSource",
                     fileNames = cms.untracked.vstring(
'file:/localdatadisk/data/MiniDAQ/closed/run73698_CMSSW_2_1_9_merge_0.root'
     )
   )
