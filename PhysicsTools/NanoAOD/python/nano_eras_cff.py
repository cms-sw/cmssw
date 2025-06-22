from Configuration.Eras.Modifier_run2_egamma_2016_cff import run2_egamma_2016
from Configuration.Eras.Modifier_run2_egamma_2017_cff import run2_egamma_2017
from Configuration.Eras.Modifier_run2_egamma_2018_cff import run2_egamma_2018
from Configuration.Eras.Modifier_run2_jme_2016_cff import run2_jme_2016
from Configuration.Eras.Modifier_run2_jme_2017_cff import run2_jme_2017
from Configuration.Eras.Modifier_run2_jme_2018_cff import run2_jme_2018
from Configuration.Eras.Modifier_run2_muon_2016_cff import run2_muon_2016
from Configuration.Eras.Modifier_run2_muon_2017_cff import run2_muon_2017
from Configuration.Eras.Modifier_run2_muon_2018_cff import run2_muon_2018
from Configuration.Eras.Modifier_run3_muon_cff import run3_muon

from Configuration.Eras.Modifier_run2_HLTconditions_2016_cff import run2_HLTconditions_2016
from Configuration.Eras.Modifier_run2_HLTconditions_2017_cff import run2_HLTconditions_2017
from Configuration.Eras.Modifier_run2_HLTconditions_2018_cff import run2_HLTconditions_2018

from Configuration.Eras.Modifier_run2_nanoAOD_106Xv2_cff import run2_nanoAOD_106Xv2
from Configuration.Eras.Modifier_tracker_apv_vfp30_2016_cff import tracker_apv_vfp30_2016

from Configuration.Eras.Modifier_run3_common_cff import run3_common
from Configuration.Eras.Modifier_run3_nanoAOD_pre142X_cff import run3_nanoAOD_pre142X
from Configuration.Eras.Modifier_run3_jme_Winter22runsBCDEprompt_cff import run3_jme_Winter22runsBCDEprompt
from Configuration.Eras.Modifier_run3_nanoAOD_2025_cff import run3_nanoAOD_2025  # for 2025 data-taking (and possibly also 2026)
from Configuration.Eras.Modifier_run3_nanoAOD_devel_cff import run3_nanoAOD_devel  # for development beyond v15

from Configuration.ProcessModifiers.nanoAOD_rePuppi_cff import nanoAOD_rePuppi

# [General Note]
# use `runX_nanoAOD_YYY` only for input-dataset-specific changes
# (e.g., run2_nanoAOD_106Xv2 for 106X MiniAODv2, run3_nanoAOD_pre142X for pre-142X Run3 MiniAODs)
run2_nanoAOD_ANY = (
    run2_nanoAOD_106Xv2
)

# use other modifiers for intrinsic era-dependent changes
run2_egamma = (run2_egamma_2016 | run2_egamma_2017 | run2_egamma_2018)
run2_muon = (run2_muon_2016 | run2_muon_2017 | run2_muon_2018)
