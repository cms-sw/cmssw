import FWCore.ParameterSet.Config as cms
from Configuration.Eras.Modifier_phase2_hgcalV19_cff import phase2_hgcalV19

# Number of silicon thickness/type categories:
# <V19: HD120, LD200, LD300
# >=V19: HD120, LD200, LD300, HD200
HGCAL_reco_constants = cms.PSet(
    numberOfThicknesses = cms.uint32(3),
    maxNumberOfThickIndices = cms.uint32(6),
)

phase2_hgcalV19.toModify(HGCAL_reco_constants,
                         numberOfThicknesses = 4,
                         maxNumberOfThickIndices = 8,
                         )
