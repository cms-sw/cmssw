import FWCore.ParameterSet.Config as cms

# L1Extra versus Reco DQM 
#     
# V.M. Ghete - HEPHY Vienna - 2011-01-03 

from DQM.L1TMonitor.L1ExtraInputTagSet_cff import *


l1ExtraRecoDQM = cms.EDAnalyzer("L1ExtraRecoDQM",
                            
    # L1Extra input tags
    L1ExtraInputTagSet,
    # reconstructed objects: collections, input tags defined as for L1Extra in a cff file
    # to be able include exactly the same objects for Validation/L1Trigger FIXME
    #
    DQMStore = cms.untracked.bool(True),
    DirName=cms.untracked.string("L1T/L1ExtraReco"),
    disableROOToutput = cms.untracked.bool(True),
    verbose = cms.untracked.bool(False),
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

