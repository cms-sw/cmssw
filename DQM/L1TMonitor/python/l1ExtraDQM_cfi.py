import FWCore.ParameterSet.Config as cms

# L1Extra DQM 
#     
# V.M. Ghete 2010-02-27

from DQM.L1TMonitor.L1ExtraInputTagSet_cff import *


l1ExtraDQM = cms.EDAnalyzer("L1ExtraDQM",
                            
    # L1Extra input tags
    L1ExtraInputTagSet,
    #
    L1ExtraIsoTauJetSource_ = cms.InputTag("fake"),
    DQMStore = cms.untracked.bool(True),
    #DirName=cms.untracked.string("L1T/L1Extra"),
    DirName=cms.string("L1T/L1Extra"),
    disableROOToutput = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False),
    #stage1_layer2_ = cms.untracked.bool(False),
    stage1_layer2_ = cms.bool(False),
    #
    #
    # number of BxInEvent for GCT and GMT  
    #  0 - zero BxInEvent 
    #  1 - central BxInEvent, (BX 0 - bunch cross with L1A)
    #  3 - (-1, 0, +1)
    #  5 - (-2, -1, 0, +1, 2)
    # -1 - take it from event setup FIXME NOT IMPLEMENTED YET
    NrBxInEventGmt = cms.int32(5),
    NrBxInEventGct = cms.int32(5)
    
)

