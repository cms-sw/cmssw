import FWCore.ParameterSet.Config as cms

# L1 Trigger DQM sequence for offline DQM
#
# used by DQM GUI: DQM/Configuration 
#
#
#
# standard RawToDigi sequence and RECO sequence must be run before the L1 Trigger modules, 
# labels from the standard sequence are used as default for the L1 Trigger DQM modules
#
# V.M. Ghete - HEPHY Vienna - 2011-01-02 
#                       
                      

#
# DQM L1 Trigger in offline environment
#

import DQMServices.Components.DQMEnvironment_cfi
dqmEnvL1T = DQMServices.Components.DQMEnvironment_cfi.dqmEnv.clone()
dqmEnvL1T.subSystemFolder = 'L1T'

# DQM online L1 Trigger modules, with offline configuration 
from DQMOffline.L1Trigger.L1TMonitorOffline_cff import *
from DQMOffline.L1Trigger.L1TMonitorClientOffline_cff import *


# DQM offline L1 Trigger versus Reco modules

import DQMServices.Components.DQMEnvironment_cfi
dqmEnvL1TriggerReco = DQMServices.Components.DQMEnvironment_cfi.dqmEnv.clone()
dqmEnvL1TriggerReco.subSystemFolder = 'L1T/L1TriggerVsReco'

#
# DQM L1 Trigger Emulator in offline environment
# Run also the L1HwVal producers (L1 Trigger emulators)
#

import DQMServices.Components.DQMEnvironment_cfi
dqmEnvL1TEMU = DQMServices.Components.DQMEnvironment_cfi.dqmEnv.clone()
dqmEnvL1TEMU.subSystemFolder = 'L1TEMU'

# DQM Offline Step 1 cfi/cff imports
from DQMOffline.L1Trigger.L1TRate_Offline_cfi import *
from DQMOffline.L1Trigger.L1TSync_Offline_cfi import *
from DQMOffline.L1Trigger.L1TEmulatorMonitorOffline_cff import *  
l1TdeRCT.rctSourceData = 'gctDigis'

# DQM Offline Step 2 cfi/cff imports
from DQMOffline.L1Trigger.L1TEmulatorMonitorClientOffline_cff import *
from DQMOffline.L1Trigger.L1TEmulatorMonitorClientOffline_cff import *


# Stage1 customization
l1TdeRCT.rctSourceData = 'gctDigis'
l1TdeRCTfromRCT.rctSourceData = 'gctDigis'
l1tRct.rctSource = 'gctDigis'
l1tRctfromRCT.rctSource = 'gctDigis'
l1tPUM.regionSource = cms.InputTag("gctDigis")

l1tStage1Layer2.gctCentralJetsSource = cms.InputTag("gctDigis","cenJets")
l1tStage1Layer2.gctForwardJetsSource = cms.InputTag("gctDigis","forJets")
l1tStage1Layer2.gctTauJetsSource = cms.InputTag("gctDigis","tauJets")
l1tStage1Layer2.gctIsoTauJetsSource = cms.InputTag("","")       
l1tStage1Layer2.gctEnergySumsSource = cms.InputTag("gctDigis")
l1tStage1Layer2.gctIsoEmSource = cms.InputTag("gctDigis","isoEm")
l1tStage1Layer2.gctNonIsoEmSource = cms.InputTag("gctDigis","nonIsoEm")
l1tStage1Layer2.stage1_layer2_ = cms.bool(False)

dqmL1ExtraParticlesStage1.etTotalSource = 'gctDigis'
dqmL1ExtraParticlesStage1.nonIsolatedEmSource = 'gctDigis:nonIsoEm'
dqmL1ExtraParticlesStage1.etMissSource = 'gctDigis'
dqmL1ExtraParticlesStage1.htMissSource = 'gctDigis'
dqmL1ExtraParticlesStage1.forwardJetSource = 'gctDigis:forJets'
dqmL1ExtraParticlesStage1.centralJetSource = 'gctDigis:cenJets'
dqmL1ExtraParticlesStage1.tauJetSource = 'gctDigis:tauJets'
dqmL1ExtraParticlesStage1.isolatedEmSource = 'gctDigis:isoEm'
dqmL1ExtraParticlesStage1.etHadSource = 'gctDigis'
dqmL1ExtraParticlesStage1.hfRingEtSumsSource = 'gctDigis'
dqmL1ExtraParticlesStage1.hfRingBitCountsSource = 'gctDigis'
l1ExtraDQMStage1.stage1_layer2_ = cms.bool(False)
l1ExtraDQMStage1.L1ExtraIsoTauJetSource_ = cms.InputTag("fake")

l1compareforstage1.GCTsourceData = cms.InputTag("gctDigis")
l1compareforstage1.GCTsourceEmul = cms.InputTag("valGctDigis")
l1compareforstage1.stage1_layer2_ = cms.bool(False)

valStage1GtDigis.GctInputTag = 'gctDigis'


from Configuration.Eras.Modifier_stage1L1Trigger_cff import stage1L1Trigger
stage1L1Trigger.toModify(l1TdeRCT, rctSourceData = 'caloStage1Digis')
stage1L1Trigger.toModify(l1TdeRCTfromRCT, rctSourceData = 'rctDigis')
stage1L1Trigger.toModify(l1tRct, rctSource = 'caloStage1Digis')
stage1L1Trigger.toModify(l1tRctfromRCT, rctSource = 'rctDigis')
stage1L1Trigger.toModify(l1tPUM, regionSource = cms.InputTag("rctDigis"))

stage1L1Trigger.toModify(l1tStage1Layer2, stage1_layer2_ = cms.bool(True))
stage1L1Trigger.toModify(l1tStage1Layer2, gctCentralJetsSource = cms.InputTag("caloStage1LegacyFormatDigis","cenJets"))
stage1L1Trigger.toModify(l1tStage1Layer2, gctForwardJetsSource = cms.InputTag("caloStage1LegacyFormatDigis","forJets"))
stage1L1Trigger.toModify(l1tStage1Layer2, gctTauJetsSource = cms.InputTag("caloStage1LegacyFormatDigis","tauJets"))
stage1L1Trigger.toModify(l1tStage1Layer2, gctIsoTauJetsSource = cms.InputTag("caloStage1LegacyFormatDigis","isoTauJets"))
stage1L1Trigger.toModify(l1tStage1Layer2, gctEnergySumsSource = cms.InputTag("caloStage1LegacyFormatDigis"))
stage1L1Trigger.toModify(l1tStage1Layer2, gctIsoEmSource = cms.InputTag("caloStage1LegacyFormatDigis","isoEm"))
stage1L1Trigger.toModify(l1tStage1Layer2, gctNonIsoEmSource = cms.InputTag("caloStage1LegacyFormatDigis","nonIsoEm"))

stage1L1Trigger.toModify( dqmL1ExtraParticlesStage1, etTotalSource = cms.InputTag("caloStage1LegacyFormatDigis") )
stage1L1Trigger.toModify( dqmL1ExtraParticlesStage1, nonIsolatedEmSource = cms.InputTag("caloStage1LegacyFormatDigis","nonIsoEm") )
stage1L1Trigger.toModify( dqmL1ExtraParticlesStage1, etMissSource = cms.InputTag("caloStage1LegacyFormatDigis") )
stage1L1Trigger.toModify( dqmL1ExtraParticlesStage1, htMissSource = cms.InputTag("caloStage1LegacyFormatDigis") )
stage1L1Trigger.toModify( dqmL1ExtraParticlesStage1, forwardJetSource = cms.InputTag("caloStage1LegacyFormatDigis","forJets") )
stage1L1Trigger.toModify( dqmL1ExtraParticlesStage1, centralJetSource = cms.InputTag("caloStage1LegacyFormatDigis","cenJets") )
stage1L1Trigger.toModify( dqmL1ExtraParticlesStage1, tauJetSource = cms.InputTag("caloStage1LegacyFormatDigis","tauJets") )
stage1L1Trigger.toModify( dqmL1ExtraParticlesStage1, isoTauJetSource = cms.InputTag("caloStage1LegacyFormatDigis","isoTauJets") )
stage1L1Trigger.toModify( dqmL1ExtraParticlesStage1, isolatedEmSource = cms.InputTag("caloStage1LegacyFormatDigis","isoEm") )
stage1L1Trigger.toModify( dqmL1ExtraParticlesStage1, etHadSource = cms.InputTag("caloStage1LegacyFormatDigis") )
stage1L1Trigger.toModify( dqmL1ExtraParticlesStage1, hfRingEtSumsSource = cms.InputTag("caloStage1LegacyFormatDigis") )
stage1L1Trigger.toModify( dqmL1ExtraParticlesStage1, hfRingBitCountsSource = cms.InputTag("caloStage1LegacyFormatDigis") )
stage1L1Trigger.toModify( l1ExtraDQMStage1, stage1_layer2_ = cms.bool(True))
stage1L1Trigger.toModify( l1ExtraDQMStage1, L1ExtraIsoTauJetSource_ = cms.InputTag("dqmL1ExtraParticlesStage1", "IsoTau"))

stage1L1Trigger.toModify(l1compareforstage1, GCTsourceData = cms.InputTag("caloStage1LegacyFormatDigis"))
stage1L1Trigger.toModify(l1compareforstage1, GCTsourceEmul = cms.InputTag("valCaloStage1LegacyFormatDigis"))
stage1L1Trigger.toModify(l1compareforstage1, stage1_layer2_ = cms.bool(True))

stage1L1Trigger.toModify(valStage1GtDigis, GctInputTag = 'caloStage1LegacyFormatDigis')

#
# define sequences 
#

l1TriggerOnline = cms.Sequence(
                               l1tMonitorStage1Online
                                * dqmEnvL1T
                               )
                                    
l1TriggerOffline = cms.Sequence(
                                l1TriggerOnline
                                 * dqmEnvL1TriggerReco
                                )
 
#
 
l1TriggerEmulatorOnline = cms.Sequence(
                                l1Stage1HwValEmulatorMonitor
                                * dqmEnvL1TEMU
                                )

l1TriggerEmulatorOffline = cms.Sequence(
                                l1TriggerEmulatorOnline                                
                                )
#

# DQM Offline Step 1 sequence
l1TriggerDqmOffline = cms.Sequence(
                                l1TriggerOffline
                                * l1tRate_Offline
                                * l1tSync_Offline
                                * l1TriggerEmulatorOffline
                                )                                  

# DQM Offline Step 2 sequence                                 
l1TriggerDqmOfflineClient = cms.Sequence(
                                l1tMonitorStage1Client
                                * l1EmulatorMonitorClient
                                )


#
#   EMERGENCY   removal of modules or full sequences 
# =============
#
# un-comment the module line below to remove the module or the sequence

#
# NOTE: for offline, remove the L1TRate which is reading from cms_orcoff_prod, but also requires 
# a hard-coded lxplus path - FIXME check if one can get rid of hard-coded path
# remove also the corresponding client
#
# L1TSync - FIXME - same problems as L1TRate


# DQM first step 
#

#l1TriggerDqmOffline.remove(l1TriggerOffline) 
#l1TriggerDqmOffline.remove(l1TriggerEmulatorOffline) 

#

#l1TriggerOffline.remove(l1TriggerOnline)


# l1tMonitorOnline sequence, defined in DQM/L1TMonitor/python/L1TMonitor_cff.py
#
#l1TriggerOnline.remove(l1tMonitorOnline)
#
l1tMonitorStage1Online.remove(bxTiming)
#l1tMonitorOnline.remove(l1tDttf)
#l1tMonitorOnline.remove(l1tCsctf) 
#l1tMonitorOnline.remove(l1tRpctf)
#l1tMonitorOnline.remove(l1tGmt)
#l1tMonitorOnline.remove(l1tGt) 
#
#l1ExtraDqmSeq.remove(dqmGctDigis)
#l1ExtraDqmSeq.remove(dqmGtDigis)
#l1ExtraDqmSeq.remove(dqmL1ExtraParticles)
#l1ExtraDqmSeq.remove(l1ExtraDQM)
#l1tMonitorOnline.remove(l1ExtraDqmSeq)
#
#l1tMonitorOnline.remove(l1tRate)
#l1tMonitorOnline.remove(l1tBPTX)
#l1tMonitorOnline.remove(l1tRctSeq)
#l1tMonitorOnline.remove(l1tGctSeq)

#

#l1TriggerEmulatorOffline.remove(l1TriggerEmulatorOnline)

# l1HwValEmulatorMonitor sequence, defined in DQM/L1TMonitor/python/L1TEmulatorMonitor_cff.py 
#
#l1TriggerEmulatorOnline.remove(l1HwValEmulatorMonitor) 

# L1HardwareValidation producers
#l1HwValEmulatorMonitor.remove(L1HardwareValidation)
#
#l1HwValEmulatorMonitor.remove(l1EmulatorMonitor)

#l1TriggerDqmOfflineClient.remove(l1tMonitorClient)
#l1TriggerDqmOfflineClient.remove(l1EmulatorMonitorClient)

# l1tMonitorClient sequence, defined in DQM/L1TMonitorClient/python/L1TMonitorClient_cff.py
#
#l1tMonitorClient.remove(l1TriggerQualityTests)
#l1tMonitorClient.remove(l1TriggerClients)

# l1TriggerClients sequence, part of l1tMonitorClient sequence

#l1TriggerClients.remove(l1tGctClient)
#l1TriggerClients.remove(l1tDttfClient)
#l1TriggerClients.remove(l1tCsctfClient) 
#l1TriggerClients.remove(l1tRpctfClient)
#l1TriggerClients.remove(l1tGmtClient)
#l1TriggerClients.remove(l1tOccupancyClient)
l1TriggerStage1Clients.remove(l1tTestsSummary)
#l1TriggerClients.remove(l1tEventInfoClient)
                              
# l1EmulatorMonitorClient sequence, defined in DQM/L1TMonitorClient/python/L1TEMUMonitorClient_cff.py
#
#l1EmulatorMonitorClient.remove(l1EmulatorQualityTests)
l1EmulatorMonitorClient.remove(l1EmulatorErrorFlagClient)
#l1EmulatorMonitorClient.remove(l1EmulatorEventInfoClient)
