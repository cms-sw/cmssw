import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.DQMOffline_Trigger_cff import *

# fix FSQ DQM - required products to calculate JEC not avaliable in HI workflow
#   this turns off JEC calculation
fsqHLTOfflineSourceSequence = cms.Sequence(fsqHLTOfflineSource)
from DQMOffline.Trigger.FSQHLTOfflineSource_cfi import getFSQHI
fsqHLTOfflineSource.todo = getFSQHI()
