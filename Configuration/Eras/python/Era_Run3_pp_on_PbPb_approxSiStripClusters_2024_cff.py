import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2024_cff import Run3_2024
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA 
from Configuration.ProcessModifiers.approxSiStripClusters_cff import approxSiStripClusters
from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3

Run3_pp_on_PbPb_approxSiStripClusters_2024 = cms.ModifierChain(Run3_2024, pp_on_AA, approxSiStripClusters, pp_on_PbPb_run3)   
