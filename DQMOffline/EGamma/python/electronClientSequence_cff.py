import FWCore.ParameterSet.Config as cms

from DQMOffline.EGamma.electronOfflineClient_cfi import *
dqmElectronClientAllElectrons = dqmElectronOfflineClient.clone() ;
dqmElectronClientAllElectrons.Selection = 0 ;
dqmElectronClientSelectionEt = dqmElectronOfflineClient.clone() ;
dqmElectronClientSelectionEt.Selection = 1 ;
dqmElectronClientSelectionEtIso = dqmElectronOfflineClient.clone() ;
dqmElectronClientSelectionEtIso.Selection = 2 ;
dqmElectronClientSelectionEtIsoElID = dqmElectronOfflineClient.clone() ;
dqmElectronClientSelectionEtIsoElID.Selection = 3 ;
dqmElectronClientTagAndProbe = dqmElectronOfflineClient.clone() ;
dqmElectronClientTagAndProbe.Selection = 4 ;

electronClientSequence = cms.Sequence(
   dqmElectronClientAllElectrons
 * dqmElectronClientSelectionEt
 * dqmElectronClientSelectionEtIso
 * dqmElectronClientSelectionEtIsoElID
 * dqmElectronClientTagAndProbe
)
