import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.trackdnn_CKF_cff import trackdnn_CKF
from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.Eras.ModifierChain_trackingMkFitProd_cff import trackingMkFitProd

Run3_noMkFit = cms.ModifierChain(Run3.copyAndExclude([trackingMkFitProd]), trackdnn_CKF)
