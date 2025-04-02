import FWCore.ParameterSet.Config as cms

# HLT Online -----------------------------------
# AlCa
#from DQM.HLTEvF.HLTAlCaMonPi0_cfi import *
#from DQM.HLTEvF.HLTAlCaMonEcalPhiSym_cfi import *
#JetMET
#from DQM.HLTEvF.HLTMonJetMETDQMSource_cfi import *
# Electron
#from DQM.HLTEvF.HLTMonEleBits_cfi import *
# Muon
#from DQM.HLTEvF.HLTMonMuonDQM_cfi import *
#from DQM.HLTEvF.HLTMonMuonBits_cfi import *
# Photon
#from DQM.HLTEvF.HLTMonPhotonBits_cfi import *
# Tau
#from DQM.HLTEvF.HLTMonTau_cfi import *
# BTag
#from DQM.HLTEvF.hltMonBTagIPSource_cfi import *
#from DQM.HLTEvF.hltMonBTagMuSource_cfi import *

# hltMonjmDQM  bombs
# hltMonMuDQM dumps names of all histograms in the directory
# hltMonPhotonBits in later releases
# *hltMonJetMET makes a log file, need to learn how to turn it off
# *hltMonEleBits causes SegmentFaults in HARVESTING(step3) in inlcuded in step2

#onlineHLTSource = cms.Sequence(EcalPi0Mon*EcalPhiSymMon*hltMonEleBits*hltMonMuBits*hltMonTauReco*hltMonBTagIPSource*hltMonBTagMuSource*dqmEnvHLTOnline)
#onlineHLTSource = cms.Sequence(EcalPi0Mon*EcalPhiSymMon*hltMonMuBits*dqmEnvHLTOnline)

# HLT Offline -----------------------------------
from DQMOffline.Trigger.dqmHLTFiltersDQMonitor_cfi import *

# EGamma
from DQMOffline.Trigger.EgHLTOfflineSource_cfi import *

# Muon
from DQMOffline.Trigger.HLTMuonOfflineAnalyzer_cosmics_cff import *

# Tau
from DQMOffline.Trigger.HLTTauDQMOffline_cff import *

# JetMET
from DQMOffline.Trigger.JetMETHLTOfflineSource_cfi import *

# Tracks
from DQMOffline.Trigger.TrackToTrackMonitoringCosmics_cff import *
from DQMOffline.Trigger.TrackingMonitoringCosmics_cff import *

import DQMServices.Components.DQMEnvironment_cfi
dqmEnvHLT= DQMServices.Components.DQMEnvironment_cfi.dqmEnv.clone(
    subSystemFolder = 'HLT',
    showHLTGlobalTag = True)

from DQMServices.Core.DQMEDAnalyzer import DQMEDAnalyzer
dqmInfoHLTMon = DQMEDAnalyzer('DQMEventInfo',
                              subSystemFolder = cms.untracked.string('HLT'),
                              showHLTGlobalTag =  cms.untracked.bool(True))

offlineHLTSource = cms.Sequence(
    cosmicTrackingMonitorHLT *
    hltToOfflineCosmicsTrackValidatorSequence *
    dqmHLTFiltersDQMonitor *
    egHLTOffDQMSource *
    hltMuonOfflineAnalyzers *
    HLTTauDQMOffline *
    jetMETHLTOfflineSource *
    dqmEnvHLT *
    dqmInfoHLTMon
)

#triggerCosmicOfflineDQMSource = cms.Sequence(onlineHLTSource*offlineHLTSource)
triggerCosmicOfflineDQMSource = cms.Sequence(offlineHLTSource)

# sequences run @tier0 on CosmicHLTMonitor PD
OfflineHLTMonitoring = cms.Sequence(
    cosmicTrackingMonitorHLT *
    hltToOfflineCosmicsTrackValidatorSequence
)
