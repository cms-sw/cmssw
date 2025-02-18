import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2024_cff import Run3_2024
from Configuration.ProcessModifiers.approxSiStripClusters_cff import approxSiStripClusters

Run3_2024_approxSiStripClusters = cms.ModifierChain(Run3_2024,  approxSiStripClusters) 
