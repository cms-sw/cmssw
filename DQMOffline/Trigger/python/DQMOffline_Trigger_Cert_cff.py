import FWCore.ParameterSet.Config as cms

#  This is  where we want to invoke both HLT and L1T certificaion
#
###################################################################

# L1T - todo
#from DQMOffline.Trigger.DQMOffline_L1T_Cert_cff import *

# HLT
from DQMOffline.Trigger.DQMOffline_HLT_Cert_cff import *

#--- Note: hltOverallCertSeq must be the last sequence!
#-- it relies on bits set in the other sequences

dqmOfflineTriggerCert = cms.Sequence(dqmOfflineHLTCert)
