import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2024_cff import Run3_2024
from Configuration.ProcessModifiers.approxSiStripClusters_cff import approxSiStripClusters
from Configuration.ProcessModifiers.trackingNoLoopers_cff import trackingNoLoopers

Run3_2024_approxSiStripClusters = cms.ModifierChain(Run3_2024.copyAndExclude([trackingNoLoopers]),  approxSiStripClusters) 
