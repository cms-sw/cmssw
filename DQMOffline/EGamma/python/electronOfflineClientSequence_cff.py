import FWCore.ParameterSet.Config as cms

from DQMOffline.EGamma.electronOfflineClient_cfi import *

dqmElectronClientAllElectrons = dqmElectronOfflineClient.clone() ;
dqmElectronClientAllElectrons.InputFolderName = cms.string("AllElectrons") ;
dqmElectronClientAllElectrons.OutputFolderName = cms.string("AllElectrons") ;

dqmElectronClientSelectionEt = dqmElectronOfflineClient.clone() ;
dqmElectronClientSelectionEt.InputFolderName = cms.string("Et10") ;
dqmElectronClientSelectionEt.OutputFolderName = cms.string("Et10") ;

dqmElectronClientSelectionEtIso = dqmElectronOfflineClient.clone() ;
dqmElectronClientSelectionEtIso.InputFolderName = cms.string("Et10Iso1") ;
dqmElectronClientSelectionEtIso.OutputFolderName = cms.string("Et10Iso1") ;

dqmElectronClientSelectionEtIsoElID = dqmElectronOfflineClient.clone() ;
dqmElectronClientSelectionEtIsoElID.InputFolderName = cms.string("Et10Iso1ElID") ;
dqmElectronClientSelectionEtIsoElID.OutputFolderName = cms.string("Et10Iso1ElID") ;

dqmElectronClientTagAndProbe = dqmElectronOfflineClient.clone() ;
dqmElectronClientTagAndProbe.InputFolderName = cms.string("TagAndProbe") ;
dqmElectronClientTagAndProbe.OutputFolderName = cms.string("TagAndProbe") ;
dqmElectronClientTagAndProbe.EffHistoTitle = cms.string("")

electronOfflineClientSequence = cms.Sequence(
   dqmElectronClientAllElectrons
 * dqmElectronClientSelectionEt
 * dqmElectronClientSelectionEtIso
# * dqmElectronClientSelectionEtIsoElID
 * dqmElectronClientTagAndProbe
)
