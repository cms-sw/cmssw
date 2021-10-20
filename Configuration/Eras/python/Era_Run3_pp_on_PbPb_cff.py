import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_noMkFit_cff import Run3_noMkFit
from Configuration.ProcessModifiers.pp_on_AA_cff import pp_on_AA
from Configuration.Eras.Modifier_pp_on_PbPb_run3_cff import pp_on_PbPb_run3
from Configuration.ProcessModifiers.trackingNoLoopers_cff import trackingNoLoopers

<<<<<<< HEAD
Run3_pp_on_PbPb = cms.ModifierChain(Run3_noMkFit.copyAndExclude([trackingNoLoopers]), pp_on_AA, pp_on_PbPb_run3)
=======
Run3_pp_on_PbPb = cms.ModifierChain(Run3_noMkFit.copyAndExclude([trackdnn, trackdnn_CKF]), pp_on_AA, pp_on_PbPb_run3)
>>>>>>> remove dnn in the heavy ion track selection
