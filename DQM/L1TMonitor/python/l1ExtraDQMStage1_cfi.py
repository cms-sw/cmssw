import FWCore.ParameterSet.Config as cms

# L1Extra DQM 
#     
# V.M. Ghete 2010-02-27

from DQM.L1TMonitor.L1ExtraInputTagSetStage1_cff import *


l1ExtraDQMStage1 = cms.EDAnalyzer("L1ExtraDQM",
                            
    # L1Extra input tags
    L1ExtraInputTagSetStage1,
    L1ExtraIsoTauJetSource_ = cms.InputTag("dqmL1ExtraParticlesStage1", "IsoTau"),
    #
    DQMStore = cms.untracked.bool(True),
    DirName=cms.string("L1T/L1ExtraStage1"),
    disableROOToutput = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False),
    stage1_layer2_ = cms.bool(True),
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

