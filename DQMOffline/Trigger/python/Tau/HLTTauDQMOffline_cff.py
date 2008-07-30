import FWCore.ParameterSet.Config as cms

#Load Electron-ID specific
#import RecoEgamma.ElectronIdentification.electronIdCutBased_cfi
#elecID = RecoEgamma.ElectronIdentification.electronIdCutBased_cfi.eidCutBased.clone()
#import RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi
#elecIDext = RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi.eidCutBasedExt.clone()
#elecID.electronQuality = 'robust'

#Load DQM
from DQMOffline.Trigger.Tau.HLTTauReferences_cfi import *
from DQMOffline.Trigger.Tau.HLTTauDQMOfflineL1Tau_cfi import *
from DQMOffline.Trigger.Tau.HLTTauDQMOfflineDoubleTau_cfi import *
from DQMOffline.Trigger.Tau.HLTTauDQMOfflineSingleTau_cfi import *
from DQMOffline.Trigger.Tau.HLTTauDQMOfflineSingleTauMET_cfi import *
from DQMOffline.Trigger.Tau.HLTTauDQMOfflineElectronTau_cfi import *
from DQMOffline.Trigger.Tau.HLTTauDQMOfflineMuonTau_cfi import *

HLTTauDQMOffline = cms.Path(TauRefProducer+HLTTauL1DQMSequence+HLTTauOfflineDQMDoubleTauSequence+HLTTauOfflineDQMSingleTauSequence+HLTTauOfflineDQMSingleTauMETSequence+HLTTauOfflineDQMElectronTauSequence+HLTTauOfflineDQMMuonTauSequence)



