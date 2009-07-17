import FWCore.ParameterSet.Config as cms

# L1 reconstruction sequence for data and MC
#     L1Extra (BX with L1A)
#     l1GtRecord - requires functional L1 O2O
#     
# V.M. Ghete 2009-07-11

 
#--- L1Extra ---#
from L1Trigger.L1ExtraFromDigis.l1extraParticles_cfi import *
l1extraParticles.centralBxOnly = False

#--- L1 GT lite record ---#
#import EventFilter.L1GlobalTriggerRawToDigi.l1GtRecord_cfi

L1Reco = cms.Sequence(l1extraParticles)
#L1Reco = cms.Sequence(l1extraParticles+l1GtRecord)
