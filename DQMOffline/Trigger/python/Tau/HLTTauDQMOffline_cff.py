import FWCore.ParameterSet.Config as cms

#Load Electron-ID specific

from RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi import eidCutBasedExt
electronIdCutBasedRobust = eidCutBasedExt.copy();electronIdCutBasedRobust.electronQuality = 'robust'

#Load DQM
from DQMOffline.Trigger.Tau.HLTTauReferences_cfi import *
from DQMOffline.Trigger.Tau.HLTTauDQMOfflineL1Tau_cfi import *
from DQMOffline.Trigger.Tau.HLTTauDQMOfflineDoubleTau_cfi import *
from DQMOffline.Trigger.Tau.HLTTauDQMOfflineSingleTau_cfi import *
from DQMOffline.Trigger.Tau.HLTTauDQMOfflineSingleTauMET_cfi import *
from DQMOffline.Trigger.Tau.HLTTauDQMOfflineElectronTau_cfi import *
from DQMOffline.Trigger.Tau.HLTTauDQMOfflineMuonTau_cfi import *

HLTTauDQMOffline = cms.Sequence(TauRefProducer+HLTTauL1DQMSequence+HLTTauOfflineDQMDoubleTauSequence+HLTTauOfflineDQMSingleTauSequence+HLTTauOfflineDQMSingleTauMETSequence+HLTTauOfflineDQMElectronTauSequence+HLTTauOfflineDQMMuonTauSequence)

#### IF WANT TO RUN ELECTRON ID ACTIVATE LINE BELOW AND DE-ACTIVATE LINE ABOVE####
#HLTTauDQMOffline = cms.Sequence(electronIdCutBasedRobust+TauRefProducer+HLTTauL1DQMSequence+HLTTauOfflineDQMDoubleTauSequence+HLTTauOfflineDQMSingleTauSequence+HLTTauOfflineDQMSingleTauMETSequence+HLTTauOfflineDQMElectronTauSequence+HLTTauOfflineDQMMuonTauSequence)


