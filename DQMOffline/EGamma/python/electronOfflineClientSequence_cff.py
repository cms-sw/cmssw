import FWCore.ParameterSet.Config as cms

from DQMOffline.EGamma.electronOfflineClient_cfi import *

dqmElectronClientAllElectrons = dqmElectronOfflineClient.clone() ;
dqmElectronClientAllElectrons.InputFolderName = cms.string("Ele2_All") ;
dqmElectronClientAllElectrons.OutputFolderName = cms.string("Ele2_All") ;

dqmElectronClientSelectionEt = dqmElectronOfflineClient.clone() ;
dqmElectronClientSelectionEt.InputFolderName = cms.string("Ele3_Et10") ;
dqmElectronClientSelectionEt.OutputFolderName = cms.string("Ele3_Et10") ;

dqmElectronClientSelectionEtIso = dqmElectronOfflineClient.clone() ;
dqmElectronClientSelectionEtIso.InputFolderName = cms.string("Ele4_Et10TkIso1") ;
dqmElectronClientSelectionEtIso.OutputFolderName = cms.string("Ele4_Et10TkIso1") ;

#dqmElectronClientSelectionEtIsoElID = dqmElectronOfflineClient.clone() ;
#dqmElectronClientSelectionEtIsoElID.InputFolderName = cms.string("Ele4_Et10TkIso1ElID") ;
#dqmElectronClientSelectionEtIsoElID.OutputFolderName = cms.string("Ele4_Et10TkIso1ElID") ;

dqmElectronClientTagAndProbe = dqmElectronOfflineClient.clone() ;
dqmElectronClientTagAndProbe.InputFolderName = cms.string("Ele5_TagAndProbe") ;
dqmElectronClientTagAndProbe.OutputFolderName = cms.string("Ele5_TagAndProbe") ;
dqmElectronClientTagAndProbe.EffHistoTitle = cms.string("")

electronOfflineClientSequence = cms.Sequence(
   dqmElectronClientAllElectrons
 * dqmElectronClientSelectionEt
 * dqmElectronClientSelectionEtIso
# * dqmElectronClientSelectionEtIsoElID
 * dqmElectronClientTagAndProbe
)
