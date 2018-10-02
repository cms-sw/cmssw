import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.JetMonitor_cfi import hltJetMETmonitoring
from DQMOffline.Trigger.HTMonitor_cfi import hltHTmonitoring
from DQMOffline.Trigger.MjjMonitor_cfi import hltMjjmonitoring
from DQMOffline.Trigger.SoftdropMonitor_cfi import hltSoftdropmonitoring
from DQMOffline.Trigger.B2GTnPMonitor_cfi import B2GegmGsfElectronIDsForDQM,B2GegHLTDQMOfflineTnPSource
from DQMOffline.Trigger.topDiLeptonHLTEventDQM_cfi import topDiLeptonHLTOfflineDQM


# B2G triggers:
# HLT_PFHT1050_v*
# HLT_AK8PFJet500_v*
# HLT_AK8PFHT750_TrimMass50_v*
# HLT_AK8PFJet380_TrimMass30_v*
# HLT_AK8PFHT800_TrimMass50_v*
# HLT_AK8PFJet400_TrimMass30_v*
# HLT_AK8PFHT850_TrimMass50_v*
# HLT_AK8PFJet420_TrimMass30_v*
# HLT_AK8PFHT900_TrimMass50_v*
# HLT_AK8PFHT700_TrimR0p1PT0p03Mass50

PFHT1050_Mjjmonitoring = hltMjjmonitoring.clone()
PFHT1050_Mjjmonitoring.FolderName = cms.string('HLT/B2G/PFHT1050')
PFHT1050_Mjjmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT1050_v*")
PFHT1050_Mjjmonitoring.jets = cms.InputTag("ak8PFJetsCHS")
PFHT1050_Mjjmonitoring.jetSelection = cms.string("pt > 200 && eta < 2.4")

PFHT1050_Softdropmonitoring = hltSoftdropmonitoring.clone()
PFHT1050_Softdropmonitoring.FolderName = cms.string('HLT/B2G/PFHT1050')
PFHT1050_Softdropmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_PFHT1050_v*")
PFHT1050_Softdropmonitoring.jetSelection = cms.string("pt > 65 && eta < 2.4")


AK8PFJet500_Mjjmonitoring = hltMjjmonitoring.clone()
AK8PFJet500_Mjjmonitoring.FolderName = cms.string('HLT/B2G/AK8PFJet500')
AK8PFJet500_Mjjmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet500_v*")
AK8PFJet500_Mjjmonitoring.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFJet500_Mjjmonitoring.jetSelection = cms.string("pt > 200 && eta < 2.4")

AK8PFJet500_Softdropmonitoring = hltSoftdropmonitoring.clone()
AK8PFJet500_Softdropmonitoring.FolderName = cms.string('HLT/B2G/AK8PFJet500')
AK8PFJet500_Softdropmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet500_v*")
AK8PFJet500_Softdropmonitoring.jetSelection = cms.string("pt > 65 && eta < 2.4")


AK8PFHT750_TrimMass50_HTmonitoring = hltHTmonitoring.clone()
AK8PFHT750_TrimMass50_HTmonitoring.FolderName = cms.string('HLT/B2G/AK8PFHT750_TrimMass50')
AK8PFHT750_TrimMass50_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFHT750_TrimMass50_v*")
AK8PFHT750_TrimMass50_HTmonitoring.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFHT750_TrimMass50_HTmonitoring.jetSelection      = cms.string("pt > 0 && eta < 2.5")
AK8PFHT750_TrimMass50_HTmonitoring.jetSelection_HT = cms.string("pt > 200 && eta < 2.5")

AK8PFHT750_TrimMass50_Mjjmonitoring = hltMjjmonitoring.clone()
AK8PFHT750_TrimMass50_Mjjmonitoring.FolderName = cms.string('HLT/B2G/AK8PFHT750_TrimMass50')
AK8PFHT750_TrimMass50_Mjjmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFHT750_TrimMass50_v*")
AK8PFHT750_TrimMass50_Mjjmonitoring.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFHT750_TrimMass50_Mjjmonitoring.jetSelection = cms.string("pt > 200 && eta < 2.4")

AK8PFHT750_TrimMass50_Softdropmonitoring = hltSoftdropmonitoring.clone()
AK8PFHT750_TrimMass50_Softdropmonitoring.FolderName = cms.string('HLT/B2G/AK8PFHT750_TrimMass50')
AK8PFHT750_TrimMass50_Softdropmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFHT750_TrimMass50_v*")
AK8PFHT750_TrimMass50_Softdropmonitoring.jetSelection = cms.string("pt > 65 && eta < 2.4")


AK8PFHT800_TrimMass50_HTmonitoring = hltHTmonitoring.clone()
AK8PFHT800_TrimMass50_HTmonitoring.FolderName = cms.string('HLT/B2G/AK8PFHT800_TrimMass50')
AK8PFHT800_TrimMass50_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFHT800_TrimMass50_v*")
AK8PFHT800_TrimMass50_HTmonitoring.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFHT800_TrimMass50_HTmonitoring.jetSelection      = cms.string("pt > 0 && eta < 2.5")
AK8PFHT800_TrimMass50_HTmonitoring.jetSelection_HT = cms.string("pt > 200 && eta < 2.5")

AK8PFHT800_TrimMass50_Mjjmonitoring = hltMjjmonitoring.clone()
AK8PFHT800_TrimMass50_Mjjmonitoring.FolderName = cms.string('HLT/B2G/AK8PFHT800_TrimMass50')
AK8PFHT800_TrimMass50_Mjjmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFHT800_TrimMass50_v*")
AK8PFHT800_TrimMass50_Mjjmonitoring.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFHT800_TrimMass50_Mjjmonitoring.jetSelection = cms.string("pt > 200 && eta < 2.4")

AK8PFHT800_TrimMass50_Softdropmonitoring = hltSoftdropmonitoring.clone()
AK8PFHT800_TrimMass50_Softdropmonitoring.FolderName = cms.string('HLT/B2G/AK8PFHT800_TrimMass50')
AK8PFHT800_TrimMass50_Softdropmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFHT800_TrimMass50_v*")
AK8PFHT800_TrimMass50_Softdropmonitoring.jetSelection = cms.string("pt > 65 && eta < 2.4")


AK8PFHT850_TrimMass50_HTmonitoring = hltHTmonitoring.clone()
AK8PFHT850_TrimMass50_HTmonitoring.FolderName = cms.string('HLT/B2G/AK8PFHT850_TrimMass50')
AK8PFHT850_TrimMass50_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFHT850_TrimMass50_v*")
AK8PFHT850_TrimMass50_HTmonitoring.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFHT850_TrimMass50_HTmonitoring.jetSelection      = cms.string("pt > 0 && eta < 2.5")
AK8PFHT850_TrimMass50_HTmonitoring.jetSelection_HT = cms.string("pt > 200 && eta < 2.5")

AK8PFHT850_TrimMass50_Mjjmonitoring = hltMjjmonitoring.clone()
AK8PFHT850_TrimMass50_Mjjmonitoring.FolderName = cms.string('HLT/B2G/AK8PFHT850_TrimMass50')
AK8PFHT850_TrimMass50_Mjjmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFHT850_TrimMass50_v*")
AK8PFHT850_TrimMass50_Mjjmonitoring.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFHT850_TrimMass50_Mjjmonitoring.jetSelection = cms.string("pt > 200 && eta < 2.4")

AK8PFHT850_TrimMass50_Softdropmonitoring = hltSoftdropmonitoring.clone()
AK8PFHT850_TrimMass50_Softdropmonitoring.FolderName = cms.string('HLT/B2G/AK8PFHT850_TrimMass50')
AK8PFHT850_TrimMass50_Softdropmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFHT850_TrimMass50_v*")
AK8PFHT850_TrimMass50_Softdropmonitoring.jetSelection = cms.string("pt > 65 && eta < 2.4")


AK8PFHT900_TrimMass50_HTmonitoring = hltHTmonitoring.clone()
AK8PFHT900_TrimMass50_HTmonitoring.FolderName = cms.string('HLT/B2G/AK8PFHT900_TrimMass50')
AK8PFHT900_TrimMass50_HTmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFHT900_TrimMass50_v*")
AK8PFHT900_TrimMass50_HTmonitoring.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFHT900_TrimMass50_HTmonitoring.jetSelection      = cms.string("pt > 0 && eta < 2.5")
AK8PFHT900_TrimMass50_HTmonitoring.jetSelection_HT = cms.string("pt > 200 && eta < 2.5")

AK8PFHT900_TrimMass50_Mjjmonitoring = hltMjjmonitoring.clone()
AK8PFHT900_TrimMass50_Mjjmonitoring.FolderName = cms.string('HLT/B2G/AK8PFHT900_TrimMass50')
AK8PFHT900_TrimMass50_Mjjmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFHT900_TrimMass50_v*")
AK8PFHT900_TrimMass50_Mjjmonitoring.jets = cms.InputTag("ak8PFJetsCHS")
AK8PFHT900_TrimMass50_Mjjmonitoring.jetSelection = cms.string("pt > 200 && eta < 2.4")

AK8PFHT900_TrimMass50_Softdropmonitoring = hltSoftdropmonitoring.clone()
AK8PFHT900_TrimMass50_Softdropmonitoring.FolderName = cms.string('HLT/B2G/AK8PFHT900_TrimMass50')
AK8PFHT900_TrimMass50_Softdropmonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFHT900_TrimMass50_v*")
AK8PFHT900_TrimMass50_Softdropmonitoring.jetSelection = cms.string("pt > 65 && eta < 2.4")



AK8PFJet360_TrimMass30_PromptMonitoring = hltJetMETmonitoring.clone()
AK8PFJet360_TrimMass30_PromptMonitoring.FolderName = cms.string('HLT/B2G/AK8PFJet360_TrimMass30')
AK8PFJet360_TrimMass30_PromptMonitoring.ptcut = cms.double(360)
AK8PFJet360_TrimMass30_PromptMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet360_TrimMass30_v*")

AK8PFJet380_TrimMass30_PromptMonitoring = hltJetMETmonitoring.clone()
AK8PFJet380_TrimMass30_PromptMonitoring.FolderName = cms.string('HLT/B2G/AK8PFJet380_TrimMass30')
AK8PFJet380_TrimMass30_PromptMonitoring.ptcut = cms.double(380)
AK8PFJet380_TrimMass30_PromptMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet380_TrimMass30_v*")

AK8PFJet400_TrimMass30_PromptMonitoring = hltJetMETmonitoring.clone()
AK8PFJet400_TrimMass30_PromptMonitoring.FolderName = cms.string('HLT/B2G/AK8PFJet400_TrimMass30')
AK8PFJet400_TrimMass30_PromptMonitoring.ptcut = cms.double(400)
AK8PFJet400_TrimMass30_PromptMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet400_TrimMass30_v*")

AK8PFJet420_TrimMass30_PromptMonitoring = hltJetMETmonitoring.clone()
AK8PFJet420_TrimMass30_PromptMonitoring.FolderName = cms.string('HLT/B2G/AK8PFJet420_TrimMass30')
AK8PFJet420_TrimMass30_PromptMonitoring.ptcut = cms.double(420)
AK8PFJet420_TrimMass30_PromptMonitoring.numGenericTriggerEventPSet.hltPaths = cms.vstring("HLT_AK8PFJet420_TrimMass30_v*")


b2gDileptonHLTOfflineDQM = topDiLeptonHLTOfflineDQM.clone()
#b2gDileptonHLTOfflineDQM.setup.directory = cms.string('HLT/B2GHLTOffline/Dileptonic/CrossTriggers')
b2gDileptonHLTOfflineDQM.setup.directory = cms.string('HLT/B2G/Dileptonic/CrossTriggers')
b2gDileptonHLTOfflineDQM.setup.triggerExtras.pathsELECMU = cms.vstring(['HLT_Mu37_Ele27_CaloIdL_MW_v','HLT_Mu27_Ele37_CaloIdL_MW_v'])
b2gDileptonHLTOfflineDQM.setup.triggerExtras.pathsDIMUON = cms.vstring([''])
b2gDileptonHLTOfflineDQM.setup.triggerExtras.pathsDIELEC = cms.vstring([''])
b2gDileptonHLTOfflineDQM.preselection.trigger.select = cms.vstring(['HLT_Mu37_Ele27_CaloIdL_MW_v','HLT_Mu27_Ele37_CaloIdL_MW_v'])

b2gDimuonHLTOfflineDQM = topDiLeptonHLTOfflineDQM.clone()
#b2gDimuonHLTOfflineDQM.setup.directory = cms.string('HLT/B2GHLTOffline/Dileptonic/Dimuon')
b2gDimuonHLTOfflineDQM.setup.directory = cms.string('HLT/B2G/Dileptonic/Dimuon')
b2gDimuonHLTOfflineDQM.setup.triggerExtras.pathsELECMU = cms.vstring([''])
b2gDimuonHLTOfflineDQM.setup.triggerExtras.pathsDIMUON = cms.vstring(['HLT_Mu37_TkMu27_v'])
b2gDimuonHLTOfflineDQM.setup.triggerExtras.pathsDIELEC = cms.vstring([''])
b2gDimuonHLTOfflineDQM.preselection.trigger.select = cms.vstring(['HLT_Mu37_TkMu27'])



b2gMonitorHLT = cms.Sequence(
    PFHT1050_Mjjmonitoring +
#    PFHT1050_Softdropmonitoring +

    AK8PFJet500_Mjjmonitoring +
#    AK8PFJet500_Softdropmonitoring +

    AK8PFHT750_TrimMass50_HTmonitoring +
    AK8PFHT750_TrimMass50_Mjjmonitoring +
#    AK8PFHT750_TrimMass50_Softdropmonitoring +

    AK8PFHT800_TrimMass50_HTmonitoring +
    AK8PFHT800_TrimMass50_Mjjmonitoring +
#    AK8PFHT800_TrimMass50_Softdropmonitoring +

    AK8PFHT850_TrimMass50_HTmonitoring +
    AK8PFHT850_TrimMass50_Mjjmonitoring +
#    AK8PFHT850_TrimMass50_Softdropmonitoring +

    AK8PFHT900_TrimMass50_HTmonitoring +
    AK8PFHT900_TrimMass50_Mjjmonitoring +
#    AK8PFHT900_TrimMass50_Softdropmonitoring +

    AK8PFJet360_TrimMass30_PromptMonitoring +
    AK8PFJet380_TrimMass30_PromptMonitoring +

    AK8PFJet400_TrimMass30_PromptMonitoring +
    AK8PFJet420_TrimMass30_PromptMonitoring +

    B2GegHLTDQMOfflineTnPSource*
    b2gDileptonHLTOfflineDQM*
    b2gDimuonHLTOfflineDQM,

    cms.Task(B2GegmGsfElectronIDsForDQM) ## unschedule execution [Use of electron VID requires this module being executed first]
)
## as reported in https://github.com/cms-sw/cmssw/issues/24444
## it turned out that all softdrop modules rely on a jet collection which is available only if the miniAOD step is run @Tier0
## ==> it is fine in the PromptReco workflow, but this collection is not available in the Express reconstruction
## in addition, it is not available in the AOD (!!!!) ==> these modules needs to be run *WithRECO* step workflow (actually w/ the miniAOD step ....)
b2gHLTDQMSourceWithRECO = cms.Sequence(
    PFHT1050_Softdropmonitoring +
    AK8PFJet500_Softdropmonitoring +
    AK8PFHT750_TrimMass50_Softdropmonitoring +
    AK8PFHT800_TrimMass50_Softdropmonitoring +    
    AK8PFHT850_TrimMass50_Softdropmonitoring +
    AK8PFHT900_TrimMass50_Softdropmonitoring    
)
b2gHLTDQMSourceExtra = cms.Sequence(
)
