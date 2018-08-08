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
from DQMOffline.L1Trigger.L1TEtSumJetOffline_cfi import *
from DQMOffline.L1Trigger.L1TEGammaOffline_cfi import *
from DQMOffline.L1Trigger.L1TTauOffline_cfi import *
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
    l1TriggerOnline *
    dqmEnvL1TriggerReco
)

#
from L1Trigger.Configuration.ValL1Emulator_cff import *

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

# Dummy sequences where a stage 2 equivalent exists
l1TriggerEgDqmOffline = cms.Sequence()
l1TriggerMuonDqmOffline = cms.Sequence()

# DQM Offline Step 2 sequence
l1TriggerDqmOfflineClient = cms.Sequence(
                                l1tMonitorStage1Client
                                * l1EmulatorMonitorClient
                                )

# Dummy sequences for legacy cosmics
l1TriggerDqmOfflineCosmics = cms.Sequence()
l1TriggerDqmOfflineCosmicsClient = cms.Sequence()

# Dummy sequences where a stage 2 equivalent exists
l1TriggerEgDqmOfflineClient = cms.Sequence()
l1TriggerMuonDqmOfflineClient = cms.Sequence()


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


##############################################################################
#stage2
##############################################################################

from Configuration.Eras.Modifier_stage2L1Trigger_cff import stage2L1Trigger

#from L1Trigger.L1TGlobal.hackConditions_cff import *
#from L1Trigger.L1TMuon.hackConditions_cff import *
#from L1Trigger.L1TCalorimeter.hackConditions_cff import *
from L1Trigger.L1TGlobal.GlobalParameters_cff import *

from DQMOffline.L1Trigger.L1TEtSumJetOffline_cfi import *
l1tEtSumJetOfflineDQMEmu.stage2CaloLayer2JetSource=cms.InputTag("valCaloStage2Layer2Digis")
l1tEtSumJetOfflineDQMEmu.stage2CaloLayer2EtSumSource=cms.InputTag("valCaloStage2Layer2Digis")

from DQMOffline.L1Trigger.L1TEGammaOffline_cfi import *
l1tEGammaOfflineDQMEmu.stage2CaloLayer2EGammaSource=cms.InputTag("valCaloStage2Layer2Digis")

from DQMOffline.L1Trigger.L1TTauOffline_cfi import *
l1tTauOfflineDQMEmu.stage2CaloLayer2TauSource=cms.InputTag("valCaloStage2Layer2Digis")

from DQMOffline.L1Trigger.L1TMuonDQMOffline_cfi import *

from DQM.L1TMonitor.L1TStage2_cff import *
from DQMOffline.L1Trigger.L1TriggerDqmOffline_SecondStep_cff import *

##Stage 2 Emulator

from DQM.L1TMonitor.L1TStage2Emulator_cff import *
from DQM.L1TMonitorClient.L1TStage2CaloLayer2DEClient_cfi import *
from DQM.L1TMonitorClient.L1TStage2MonitorClient_cff import *
# L1T monitor client sequence (system clients and quality tests)
l1TStage2EmulatorClients = cms.Sequence(
                        l1tStage2CaloLayer2DEClient
                        # l1tStage2EmulatorEventInfoClient
                        )

l1tStage2EmulatorMonitorClient = cms.Sequence(
                        # l1TStage2EmulatorQualityTests +
                        l1TStage2EmulatorClients
                        )

#
# define sequences
#

##############################################################################
# Unpacked data sequences
Stage2l1TriggerOnline = cms.Sequence(
                                l1tStage2OnlineDQM
                                * dqmEnvL1T
                               )
# Do not include the uGT online DQM module in the offline sequence
# since the large 2D histograms cause crashes at the T0.
l1tStage2OnlineDQM.remove(l1tStage2uGT)

# sequence to run for all datasets
Stage2l1TriggerOffline = cms.Sequence(
                                Stage2l1TriggerOnline #*
                                #dqmEnvL1TriggerReco
                                )

# sequence to run only for modules requiring an electron dataset
Stage2l1tEgOffline = cms.Sequence(
                                l1tEGammaOfflineDQM
                                )

# sequence to run only for modules requiring a muon dataset
Stage2l1tMuonOffline = cms.Sequence(
                                l1tEtSumJetOfflineDQM *
                                l1tTauOfflineDQM *
                                l1tMuonDQMOffline
                                )

##############################################################################
# Emulator sequences
Stage2l1TriggerEmulatorOnline = cms.Sequence(
                                 valHcalTriggerPrimitiveDigis +
                                 Stage2L1HardwareValidation +
                                 l1tStage2EmulatorOnlineDQM +
                                 dqmEnvL1TEMU
                                )
# Do not include the uGT emulation online DQM module in the offline
# sequence since the large 2D histograms cause crashes at the T0.
l1tStage2EmulatorOnlineDQM.remove(l1tStage2uGtEmul)

# sequence to run for all datasets
Stage2l1TriggerEmulatorOffline = cms.Sequence(
                                Stage2l1TriggerEmulatorOnline
                                )

# sequence to run only for modules requiring an electron dataset
Stage2l1tEgEmulatorOffline = cms.Sequence(
                                #l1tEGammaOfflineDQMEmu
                                )

# sequence to run only for modules requiring a muon dataset
Stage2l1tMuonEmulatorOffline = cms.Sequence(
                                #l1tEtSumJetOfflineDQMEmu +
                                #l1tTauOfflineDQMEmu
                                )

##############################################################################
# DQM sequences for step 1

# DQM Offline sequence
Stage2l1TriggerDqmOffline = cms.Sequence(
                                Stage2l1TriggerOffline
                                * Stage2l1TriggerEmulatorOffline
                                )

# DQM Offline sequence for modules requiring an electron dataset
Stage2l1tEgDqmOffline = cms.Sequence(
                                Stage2l1tEgOffline
                                * Stage2l1tEgEmulatorOffline
                                )

# DQM Offline sequence for modules requiring a muon dataset
Stage2l1tMuonDqmOffline = cms.Sequence(
                                Stage2l1tMuonOffline
                                * Stage2l1tMuonEmulatorOffline
                                )

##############################################################################
# DQM sequences for step 2

# DQM Offline sequence
Stage2l1TriggerDqmOfflineClient = cms.Sequence(
                                l1tStage2EmulatorMonitorClient *
                                l1tStage2MonitorClient *
                                DQMHarvestL1TMon
                                )

# DQM Offline sequence for modules requiring an electron dataset
Stage2l1tEgDqmOfflineClient = cms.Sequence(
                                DQMHarvestL1TEg
                                )

# DQM Offline sequence for modules requiring a muon dataset
Stage2l1tMuonDqmOfflineClient = cms.Sequence(
                                DQMHarvestL1TMuon
                                )


#replacements for stage2
stage2L1Trigger.toReplaceWith(l1TriggerOnline, Stage2l1TriggerOnline)
stage2L1Trigger.toReplaceWith(l1TriggerOffline, Stage2l1TriggerOffline)
stage2L1Trigger.toReplaceWith(l1TriggerEmulatorOnline, Stage2l1TriggerEmulatorOnline)
stage2L1Trigger.toReplaceWith(l1TriggerEmulatorOffline, Stage2l1TriggerEmulatorOffline)
stage2L1Trigger.toReplaceWith(l1TriggerDqmOffline, Stage2l1TriggerDqmOffline)
stage2L1Trigger.toReplaceWith(l1TriggerEgDqmOffline, Stage2l1tEgDqmOffline)
stage2L1Trigger.toReplaceWith(l1TriggerMuonDqmOffline, Stage2l1tMuonDqmOffline)
stage2L1Trigger.toReplaceWith(l1TriggerDqmOfflineClient, Stage2l1TriggerDqmOfflineClient)
stage2L1Trigger.toReplaceWith(l1TriggerEgDqmOfflineClient, Stage2l1tEgDqmOfflineClient)
stage2L1Trigger.toReplaceWith(l1TriggerMuonDqmOfflineClient, Stage2l1tMuonDqmOfflineClient)
stage2L1Trigger.toReplaceWith(l1EmulatorMonitorClient,l1tStage2EmulatorMonitorClient)
stage2L1Trigger.toReplaceWith(l1TriggerDqmOfflineCosmics, Stage2l1TriggerDqmOffline)
stage2L1Trigger.toReplaceWith(l1TriggerDqmOfflineCosmicsClient, Stage2l1TriggerDqmOfflineClient)
