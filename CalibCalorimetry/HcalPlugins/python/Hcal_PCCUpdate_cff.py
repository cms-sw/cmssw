import FWCore.ParameterSet.Config as cms

# Turn on the PCC update which synchronize timePhase in PCC as in SIM
PCCUpdate = cms.PSet(
  applyFixPCC = cms.bool(False)
)

from Configuration.Eras.Modifier_run3_common_cff import run3_common
run3_common.toModify(PCCUpdate, applyFixPCC = True)

from Configuration.Eras.Modifier_phase2_common_cff import phase2_common
phase2_common.toModify(PCCUpdate, applyFixPCC = True)
