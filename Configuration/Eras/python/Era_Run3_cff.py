import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
from Configuration.Eras.Modifier_run3_common_cff import run3_common
from Configuration.Eras.Modifier_run3_GEM_cff import run3_GEM
from Configuration.Eras.Modifier_run3_HB_cff import run3_HB
from Configuration.Eras.Modifier_run3_HFSL_cff import run3_HFSL
from Configuration.Eras.Modifier_run2_GEM_2017_cff import run2_GEM_2017
from Configuration.Eras.Modifier_stage2L1Trigger_2021_cff import stage2L1Trigger_2021
from Configuration.Eras.Modifier_ctpps_2018_cff import ctpps_2018
from Configuration.Eras.Modifier_ctpps_2022_cff import ctpps_2022
from Configuration.ProcessModifiers.dd4hep_cff import dd4hep
from Configuration.Eras.Modifier_run3_egamma_cff import run3_egamma
from Configuration.Eras.Modifier_run2_egamma_2018_cff import run2_egamma_2018
from Configuration.Eras.Modifier_run2_HLTconditions_2018_cff import run2_HLTconditions_2018
from Configuration.Eras.Modifier_run3_RPC_cff import run3_RPC

Run3 = cms.ModifierChain(Run2_2018.copyAndExclude([run2_GEM_2017, ctpps_2018, run2_egamma_2018, run2_HLTconditions_2018]),
                         run3_common, run3_egamma, run3_GEM, run3_HB, run3_HFSL, stage2L1Trigger_2021, ctpps_2022, dd4hep, run3_RPC)

