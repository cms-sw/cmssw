import FWCore.ParameterSet.Config as cms

# Histogram limits
from L1TriggerOffline.L1Analyzer.HistoLimits_cfi import *
# Root output file
from L1TriggerOffline.L1Analyzer.TFile_cfi import *

demo = cms.EDAnalyzer('TriggerOperation',
     histoLimits,                
    L1GtReadoutRecordTag = cms.untracked.InputTag("gtDigis")                      
 ##    bitsNBins = cms.untracked.int32(128),
 ##    bitsMin = cms.untracked.double(-0.5),
 ##    bitsMax = cms.untracked.double(127.5)
              

)
