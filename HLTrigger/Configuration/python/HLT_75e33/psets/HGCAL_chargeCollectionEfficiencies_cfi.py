import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_phase2_hgcalV19_cff import phase2_hgcalV19

HGCAL_chargeCollectionEfficiencies = cms.PSet(
    values = cms.vdouble(1.0, 1.0, 1.0)
)

phase2_hgcalV19.toModify(HGCAL_chargeCollectionEfficiencies, values =  [1.0, 1.0,1.0,1.0])
