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

#import DQMServices.Components.DQMEnvironment_cfi
#dqmEnvHLTOnline = DQMServices.Components.DQMEnvironment_cfi.dqmEnv.clone()
#dqmEnvHLTOnline.subSystemFolder = 'HLT'

#onlineHLTSource = cms.Sequence(EcalPi0Mon*EcalPhiSymMon*hltMonEleBits*hltMonMuBits*hltMonTauReco*hltMonBTagIPSource*hltMonBTagMuSource*dqmEnvHLTOnline)
#onlineHLTSource = cms.Sequence(EcalPi0Mon*EcalPhiSymMon*hltMonMuBits*dqmEnvHLTOnline)

# HLT Offline -----------------------------------
from DQMOffline.Trigger.HLTGeneralOffline_cfi import *

# EGamma
from DQMOffline.Trigger.EgHLTOfflineSource_cfi import *

# Muon
from DQMOffline.Trigger.HLTMuonOfflineAnalyzer_cosmics_cff import *

# Tau
from DQMOffline.Trigger.HLTTauDQMOffline_cff import *

# JetMET
from DQMOffline.Trigger.JetMETHLTOfflineSource_cfi import *

import DQMServices.Components.DQMEnvironment_cfi
dqmEnvHLT= DQMServices.Components.DQMEnvironment_cfi.dqmEnv.clone()
dqmEnvHLT.subSystemFolder = 'HLT'

offlineHLTSource = cms.Sequence(
    hltResults *
    egHLTOffDQMSource *
    hltMuonOfflineAnalyzers *
    HLTTauDQMOffline *
    jetMETHLTOfflineSource *
    dqmEnvHLT
)

#triggerCosmicOfflineDQMSource =  cms.Sequence(onlineHLTSource*offlineHLTSource)
triggerCosmicOfflineDQMSource =  cms.Sequence(offlineHLTSource)
