import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.trackdnn_CKF_cff import trackdnn_CKF
from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
from Configuration.Eras.ModifierChain_trackingMkFitProd_cff import trackingMkFitProd

Run2_2017_noMkFit = cms.ModifierChain(Run2_2017.copyAndExclude([trackingMkFitProd]), trackdnn_CKF)
