import FWCore.ParameterSet.Config as cms

from DQMOffline.EGamma.electronOfflineClient_cfi import *

dqmElectronClientAllElectrons = dqmElectronOfflineClient.clone(
    InputFolderName = "Egamma/Electrons/Ele2_All",
    OutputFolderName = "Egamma/Electrons/Ele2_All"
)
dqmElectronClientAllElectronsHGC = dqmElectronOfflineClient.clone(
    InputFolderName = "Egamma/Electrons/Ele2HGC_All",
    OutputFolderName = "Egamma/Electrons/Ele2HGC_All"
)
dqmElectronClientSelectionEt = dqmElectronOfflineClient.clone(
    InputFolderName = "Egamma/Electrons/Ele3_Et10",
    OutputFolderName = "Egamma/Electrons/Ele3_Et10"
)
dqmElectronClientSelectionEtIso = dqmElectronOfflineClient.clone(
    InputFolderName = "Egamma/Electrons/Ele4_Et10TkIso1",
    OutputFolderName = "Egamma/Electrons/Ele4_Et10TkIso1"
)
#dqmElectronClientSelectionEtIsoElID = dqmElectronOfflineClient.clone(
#InputFolderName = "Egamma/Electrons/Ele4_Et10TkIso1ElID",
#OutputFolderName = "Egamma/Electrons/Ele4_Et10TkIso1ElID"
#)
dqmElectronClientTagAndProbe = dqmElectronOfflineClient.clone(
    InputFolderName = "Egamma/Electrons/Ele5_TagAndProbe",
    OutputFolderName = "Egamma/Electrons/Ele5_TagAndProbe",
    EffHistoTitle = ""
)
electronOfflineClientSequence = cms.Sequence(
   dqmElectronClientAllElectrons
 * dqmElectronClientSelectionEt
 * dqmElectronClientSelectionEtIso
# * dqmElectronClientSelectionEtIsoElID
 * dqmElectronClientTagAndProbe
)
_electronOfflineClientSequenceHGC = electronOfflineClientSequence.copy()
_electronOfflineClientSequenceHGC += dqmElectronClientAllElectronsHGC

from Configuration.Eras.Modifier_phase2_hgcal_cff import phase2_hgcal
phase2_hgcal.toReplaceWith(
  electronOfflineClientSequence, _electronOfflineClientSequenceHGC
)

