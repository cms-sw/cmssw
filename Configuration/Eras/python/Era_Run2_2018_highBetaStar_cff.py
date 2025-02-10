import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.trackdnn_cff import trackdnn
from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
from Configuration.Eras.Modifier_highBetaStar_cff import highBetaStar

Run2_2018_highBetaStar = cms.ModifierChain(Run2_2018.copyAndExclude([trackdnn]), highBetaStar)
