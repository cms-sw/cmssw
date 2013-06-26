import FWCore.ParameterSet.Config as cms

# L1 Trigger versus Reco offline DQM sequence
#
#
# standard RawToDigi sequence and RECO sequence must be run before the L1 Trigger modules, 
# labels from the standard sequence are used as default for the L1 Trigger DQM modules
#
# V.M. Ghete - HEPHY Vienna - 2011-01-02 
#                       


#
# DQM modules
#

# L1Extra vs RECO DQM module
from DQMOffline.L1Trigger.L1ExtraRecoDQM_cff import *



#
# define sequences 
#


# L1T versus Reco offline monitor sequence 
#     modules are independent, so the order is irrelevant 

l1TriggerRecoDQM = cms.Sequence(
                          l1ExtraRecoDQM
                          )


# sequence for L1 Trigger DQM modules on EndPath 

                            

