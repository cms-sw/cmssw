import FWCore.ParameterSet.Config as cms

from DQMOffline.Trigger.JetMonitor_cfi import hltJetMETmonitoring
from DQMOffline.Trigger.HTMonitor_cfi import hltHTmonitoring
from DQMOffline.Trigger.MjjMonitor_cfi import hltMjjmonitoring
from DQMOffline.Trigger.SoftdropMonitor_cfi import hltSoftdropmonitoring
from DQMOffline.Trigger.B2GTnPMonitor_cfi import B2GegmGsfElectronIDsForDQM,B2GegHLTDQMOfflineTnPSource
from DQMOffline.Trigger.TopMonitor_cfi import hltTOPmonitoring

### B2G triggers:
# HLT_AK8PFHT*_TrimMass50
# HLT_AK8PFJet*_TrimMass30
# HLT_AK8PFJet*_MassSD30

# HLT_Mu37_Ele27_CaloIdL_MW
# HLT_Mu27_Ele37_CaloIdL_MW
# HLT_Mu37_TkMu27
#
# Additionally, we monitor mjj and mSD for PFHT1050 and AK8PFJet500

# HT and AK8jet monitoring

PFHT1050_Mjjmonitoring = hltMjjmonitoring.clone(
    FolderName = 'HLT/B2G/PFHT1050',
    jets = "ak8PFJetsPuppi",
    jetSelection = "pt > 200 && eta < 2.4",
    numGenericTriggerEventPSet= dict(hltPaths = ["HLT_PFHT1050_v*"])     
)

PFHT1050_Softdropmonitoring = hltSoftdropmonitoring.clone(
    FolderName = 'HLT/B2G/PFHT1050',
    jetSelection = "pt > 65 && eta < 2.4",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_PFHT1050_v*"])        
)

AK8PFJet500_Mjjmonitoring = hltMjjmonitoring.clone(
    FolderName = 'HLT/B2G/AK8PFJet500',
    jets = "ak8PFJetsPuppi",
    jetSelection = "pt > 200 && eta < 2.4",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJet500_v*"])
)

AK8PFJet500_Softdropmonitoring = hltSoftdropmonitoring.clone(
    FolderName = 'HLT/B2G/AK8PFJet500',
    jetSelection = "pt > 65 && eta < 2.4",
    numGenericTriggerEventPSet= dict(hltPaths = ["HLT_AK8PFJet500_v*"])
)

# AK8PFHT800_TrimMass50 monitoring

AK8PFHT800_TrimMass50_HTmonitoring = hltHTmonitoring.clone(
    FolderName = 'HLT/B2G/AK8PFHT800_TrimMass50',
    jets = "ak8PFJetsPuppi",
    jetSelection      = "pt > 0 && eta < 2.5",
    jetSelection_HT = "pt > 200 && eta < 2.5",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFHT800_TrimMass50_v*"])
)

AK8PFHT800_TrimMass50_Mjjmonitoring = hltMjjmonitoring.clone(
    FolderName = 'HLT/B2G/AK8PFHT800_TrimMass50',
    jets = "ak8PFJetsPuppi",
    jetSelection = "pt > 200 && eta < 2.4",
    numGenericTriggerEventPSet= dict(hltPaths = ["HLT_AK8PFHT800_TrimMass50_v*"])
)

AK8PFHT800_TrimMass50_Softdropmonitoring = hltSoftdropmonitoring.clone(
    FolderName = 'HLT/B2G/AK8PFHT800_TrimMass50',
    jetSelection = "pt > 65 && eta < 2.4",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFHT800_TrimMass50_v*"])
)

# AK8PFJet400_TrimMass30 monitoring

AK8PFJet400_TrimMass30_PromptMonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/B2G/AK8PFJet400_TrimMass30',
    ptcut = 400,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJet400_TrimMass30_v*"])
)

AK8PFJet400_TrimMass30_Mjjmonitoring = hltMjjmonitoring.clone(
    FolderName = 'HLT/B2G/AK8PFJet400_TrimMass30',
    jets = "ak8PFJetsPuppi",
    jetSelection = "pt > 200 && eta < 2.4",
    numGenericTriggerEventPSet= dict(hltPaths = ["HLT_AK8PFJet400_TrimMass30_v*"])
)

AK8PFJet400_TrimMass30_Softdropmonitoring = hltSoftdropmonitoring.clone(
    FolderName = 'HLT/B2G/AK8PFJet400_TrimMass30',
    jetSelection = "pt > 200 && eta < 2.4",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJet400_TrimMass30_v*"]),
    histoPSet = dict(
        htBinning = [0., 10., 20., 30., 40., 50.,  60., 70., 80., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180., 190., 200., 210., 220., 230., 240., 250., 260., 270., 280., 290., 300., 310., 320., 330., 340., 350.],
        htPSet = dict(nbins = 200, xmin = -0.5, xmax = 19999.5)
    )
)

# AK8PFJet400_MassSD30 monitoring

AK8PFJet400_MassSD30_PromptMonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/B2G/AK8PFJet400_MassSD30',
    ptcut = 400,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJet400_MassSD30_v*"])
)

AK8PFJet400_MassSD30_Mjjmonitoring = hltMjjmonitoring.clone(
    FolderName = 'HLT/B2G/AK8PFJet400_MassSD30',
    jets = "ak8PFJetsPuppi",
    jetSelection = "pt > 200 && eta < 2.4",
    numGenericTriggerEventPSet= dict(hltPaths = ["HLT_AK8PFJet400_MassSD30_v*"])
)

AK8PFJet400_MassSD30_Softdropmonitoring = hltSoftdropmonitoring.clone(
    FolderName = 'HLT/B2G/AK8PFJet400_MassSD30',
    jetSelection = "pt > 200 && eta < 2.4",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8PFJet400_MassSD30_v*"]),
    histoPSet = dict(
        htBinning = [0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180., 190., 200., 210., 220., 230., 240., 250., 260., 270., 280., 290., 300., 310., 320., 330., 340., 350.],
        htPSet = dict(nbins = 200, xmin = -0.5, xmax = 19999.5)
    )
)

# AK8DiPFJet250_250_MassSD30 monitoring

AK8DiPFJet250_250_MassSD30_PromptMonitoring = hltJetMETmonitoring.clone(
    FolderName = 'HLT/B2G/AK8DiPFJet250_250_MassSD30',
    ptcut = 400,
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8DiPFJet250_250_MassSD30_v*"])
)

AK8DiPFJet250_250_MassSD30_Mjjmonitoring = hltMjjmonitoring.clone(
    FolderName = 'HLT/B2G/AK8DiPFJet250_250_MassSD30',
    jets = "ak8PFJetsPuppi",
    jetSelection = "pt > 200 && eta < 2.4",
    numGenericTriggerEventPSet= dict(hltPaths = ["HLT_AK8DiPFJet250_250_MassSD30_v*"])
)

AK8DiPFJet250_250_MassSD30_Softdropmonitoring = hltSoftdropmonitoring.clone(
    FolderName = 'HLT/B2G/AK8DiPFJet250_250_MassSD30',
    jetSelection = "pt > 200 && eta < 2.4",
    numGenericTriggerEventPSet = dict(hltPaths = ["HLT_AK8DiPFJet250_250_MassSD30_v*"]),
    histoPSet = dict(
        htBinning = [0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130., 140., 150., 160., 170., 180., 190., 200., 210., 220., 230., 240., 250., 260., 270., 280., 290., 300., 310., 320., 330., 340., 350.],
        htPSet = dict(nbins = 200, xmin = -0.5, xmax = 19999.5)
    )
)

# Lepton cross trigger monitoring

hltDQMonitorB2G_MuEle = hltTOPmonitoring.clone(
    FolderName = 'HLT/B2G/Dileptonic/HLT_MuXX_EleXX_CaloIdL_MW',
    nelectrons = 1,
    eleSelection = 'pt>20 & abs(eta)<2.5',
    nmuons = 1,
    muoSelection = 'pt>20 & abs(eta)<2.4 & ((pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25)  & isPFMuon & (isTrackerMuon || isGlobalMuon)',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu37_Ele27_CaloIdL_MW_v*', 'HLT_Mu27_Ele37_CaloIdL_MW_v*'])
)

hltDQMonitorB2G_MuTkMu = hltTOPmonitoring.clone(
    FolderName = 'HLT/B2G/Dileptonic/HLT_Mu37_TkMu27',
    nmuons = 2,
    muoSelection = 'pt>20 & abs(eta)<2.4 & ((pfIsolationR04.sumChargedHadronPt + max(pfIsolationR04.sumNeutralHadronEt + pfIsolationR04.sumPhotonEt - (pfIsolationR04.sumPUPt)/2.,0.))/pt < 0.25)  & isPFMuon & (isTrackerMuon || isGlobalMuon)',
    numGenericTriggerEventPSet = dict(hltPaths = ['HLT_Mu37_TkMu27_v*'])
)


# the sequence

b2gMonitorHLT = cms.Sequence(

    PFHT1050_Mjjmonitoring +

    AK8PFJet500_Mjjmonitoring +

    AK8PFHT800_TrimMass50_HTmonitoring +
    AK8PFHT800_TrimMass50_Mjjmonitoring +

    AK8PFJet400_TrimMass30_PromptMonitoring +
    AK8PFJet400_TrimMass30_Mjjmonitoring +

    AK8PFJet400_MassSD30_PromptMonitoring +
    AK8PFJet400_MassSD30_Mjjmonitoring +

    AK8DiPFJet250_250_MassSD30_PromptMonitoring +
    AK8DiPFJet250_250_MassSD30_Mjjmonitoring +

    B2GegHLTDQMOfflineTnPSource

  * hltDQMonitorB2G_MuEle
  * hltDQMonitorB2G_MuTkMu

  , cms.Task(B2GegmGsfElectronIDsForDQM) ## unschedule execution [Use of electron VID requires this module being executed first]
)

## as reported in https://github.com/cms-sw/cmssw/issues/24444
## it turned out that all softdrop modules rely on a jet collection which is available only if the miniAOD step is run @Tier0
## ==> it is fine in the PromptReco workflow, but this collection is not available in the Express reconstruction
## in addition, it is not available in the AOD (!!!!) ==> these modules needs to be run *WithRECO* step workflow (actually w/ the miniAOD step ....)
b2gHLTDQMSourceWithRECO = cms.Sequence(
    PFHT1050_Softdropmonitoring +
    AK8PFJet500_Softdropmonitoring +
    AK8PFHT800_TrimMass50_Softdropmonitoring +
    AK8PFJet400_TrimMass30_Softdropmonitoring +
    AK8PFJet400_MassSD30_Softdropmonitoring +
    AK8DiPFJet250_250_MassSD30_Softdropmonitoring
)

b2gHLTDQMSourceExtra = cms.Sequence(
)
