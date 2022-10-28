import FWCore.ParameterSet.Config as cms
from Configuration.ProcessModifiers.trackdnn_CKF_cff import trackdnn_CKF
from Configuration.ProcessModifiers.trackdnn_cff import trackdnn
from Configuration.Eras.Era_Run3_cff import Run3
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3

Run3_pp_on_PbPb = cms.ModifierChain(Run3.copyAndExclude([trackdnn, trackdnn_CKF]), pp_on_AA, pp_on_PbPb_run3)


