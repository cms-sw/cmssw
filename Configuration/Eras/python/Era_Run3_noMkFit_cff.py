import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.Eras.ModifierChain_trackingMkFitProd_cff import trackingMkFitProd

Run3_noMkFit = Run3.copyAndExclude([trackingMkFitProd])
