import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.trackdnn_CKF_cff import trackdnn_CKF
from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
from Configuration.Eras.ModifierChain_trackingMkFitProd_cff import trackingMkFitProd

Run2_2018_noMkFit = cms.ModifierChain(Run2_2018.copyAndExclude([trackingMkFitProd]), trackdnn_CKF)
