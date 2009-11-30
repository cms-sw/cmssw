import FWCore.ParameterSet.Config as cms

from DQMOffline.EGamma.electronOfflineClient_cfi import *
dqmElectronOfflineClient0 = dqmElectronOfflineClient.clone() ;
dqmElectronOfflineClient0.Selection = 0 ;
dqmElectronOfflineClient1 = dqmElectronOfflineClient.clone() ;
dqmElectronOfflineClient1.Selection = 1 ;
dqmElectronOfflineClient2 = dqmElectronOfflineClient.clone() ;
dqmElectronOfflineClient2.Selection = 2 ;
dqmElectronOfflineClient3 = dqmElectronOfflineClient.clone() ;
dqmElectronOfflineClient3.Selection = 3 ;
dqmElectronOfflineClient4 = dqmElectronOfflineClient.clone() ;
dqmElectronOfflineClient4.Selection = 4 ;

electronClientSequence = cms.Sequence(dqmElectronOfflineClient0*dqmElectronOfflineClient1*dqmElectronOfflineClient2*dqmElectronOfflineClient3*dqmElectronOfflineClient4)
