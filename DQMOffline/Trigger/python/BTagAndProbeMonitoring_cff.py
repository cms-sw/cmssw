import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.BTagAndProbeMonitor_cfi import BTagAndProbeMonitoring

from Configuration.Eras.Modifier_run2_HLTconditions_2018_cff import run2_HLTconditions_2018
from Configuration.Eras.Modifier_run2_HLTconditions_2017_cff import run2_HLTconditions_2017
from Configuration.Eras.Modifier_run2_HLTconditions_2016_cff import run2_HLTconditions_2016

###
### Ele+Jet
###

BTagAndProbe_1e1m = BTagAndProbeMonitoring.clone(
    FolderName = cms.string('HLT/BTV/TnP/oneEle_oneMu'),
    nmuons = 1,
    nelectrons = 1,
    nbjets = 2,
    eleSelection = 'pt>10 & abs(eta)<2.5',
    muoSelection = 'pt>10 & abs(eta)<2.4',
    bjetSelection = 'pt>20 & abs(eta)<2.4',
    genericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12_DoublePFJets40_PFBTagDeepCSV_p71_v*', 
                                      'HLT_Mu12_DoublePFJets40MaxDeta1p6_DoublePFBTagDeepCSV_p71_v*']),
    #denGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu12_DoublePFJets40_PFBTagDeepCSV_p71_v*',
    #                                  'HLT_Mu12_DoublePFJets40MaxDeta1p6_DoublePFBTagDeepCSV_p71_v*']),
    debug = cms.bool(True),
)

### ---

BTagAndProbe_1e0m = BTagAndProbe_1e1m.clone(
    FolderName = cms.string('HLT/BTV/TnP/OneEle_NoMu'),
    nmuons = 0,
    nelectrons = 1,
    debug = False,
)

### ---
BTagAndProbe_0e1m = BTagAndProbe_1e1m.clone(
    FolderName = cms.string('HLT/BTV/TnP/NoEle_OneMu'),
    nmuons = 1,
    nelectrons = 0,
    debug = False,
)


from DQMOffline.Trigger.HLTEGTnPMonitor_cfi import egmGsfElectronIDsForDQM

BTagAndProbeHLT = cms.Sequence(

      BTagAndProbe_1e1m
    + BTagAndProbe_1e0m
    + BTagAndProbe_0e1m

    , cms.Task(egmGsfElectronIDsForDQM) # Use of electron VID requires this module being executed first
)
