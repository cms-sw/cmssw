import FWCore.ParameterSet.Config as cms

from Geometry.HcalEventSetup.hcalTopologyIdealBase_cfi import hcalTopologyIdealBase as hcalTopologyIdeal

from Configuration.Eras.Modifier_run2_HEPlan1_2017_cff import run2_HEPlan1_2017
run2_HEPlan1_2017.toModify(hcalTopologyIdeal,
    MergePosition = cms.untracked.bool(True)
)

from Configuration.ProcessModifiers.run2_HECollapse_2018_cff import run2_HECollapse_2018
run2_HECollapse_2018.toModify(hcalTopologyIdeal,
    MergePosition = cms.untracked.bool(True)
)
