import FWCore.ParameterSet.Config as cms

from DQMOffline.EGamma.electronOfflineClient_cfi import *
dqmElectronClientAllElectrons = dqmElectronOfflineClient.clone() ;
dqmElectronClientAllElectrons.Selection = 0 ;
dqmElectronClientAllElectrons.InputFolderName = cms.string("AllElectrons") ;
dqmElectronClientAllElectrons.OutputFolderName = cms.string("AllElectrons") ;
dqmElectronClientSelectionEt = dqmElectronOfflineClient.clone() ;
dqmElectronClientSelectionEt.Selection = 1 ;
dqmElectronClientSelectionEt.InputFolderName = cms.string("Et10") ;
dqmElectronClientSelectionEt.OutputFolderName = cms.string("Et10") ;
dqmElectronClientSelectionEtIso = dqmElectronOfflineClient.clone() ;
dqmElectronClientSelectionEtIso.Selection = 2 ;
dqmElectronClientSelectionEtIso.InputFolderName = cms.string("Et10Iso5") ;
dqmElectronClientSelectionEtIso.OutputFolderName = cms.string("Et10Iso5") ;
dqmElectronClientSelectionEtIsoElID = dqmElectronOfflineClient.clone() ;
dqmElectronClientSelectionEtIsoElID.Selection = 3 ;
dqmElectronClientSelectionEtIsoElID.InputFolderName = cms.string("Et10Iso5ElID") ;
dqmElectronClientSelectionEtIsoElID.OutputFolderName = cms.string("Et10Iso5ElID") ;
dqmElectronClientTagAndProbe = dqmElectronOfflineClient.clone() ;
dqmElectronClientTagAndProbe.Selection = 4 ;
dqmElectronClientTagAndProbe.InputFolderName = cms.string("TagAndProbe") ;
dqmElectronClientTagAndProbe.OutputFolderName = cms.string("TagAndProbe") ;

electronClientSequence = cms.Sequence(
   dqmElectronClientAllElectrons
 * dqmElectronClientSelectionEt
 * dqmElectronClientSelectionEtIso
# * dqmElectronClientSelectionEtIsoElID
# * dqmElectronClientTagAndProbe
)
