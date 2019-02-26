import FWCore.ParameterSet.Config as cms

minEtaCorrection = cms.double(1.4)
maxEtaCorrection = cms.double(3.0)
hadronCorrections = cms.PSet(value = cms.vdouble(1.24, 1.24, 1.24, 1.23, 1.24, 1.25, 1.29, 1.29))
egammaCorrections = cms.PSet(value = cms.vdouble(1.00, 1.00, 1.01, 1.01, 1.02, 1.03, 1.04, 1.04))

hadronCorrections_hgcalV9 = cms.vdouble(1.28, 1.28, 1.24, 1.19, 1.17, 1.17, 1.17, 1.17)
egammaCorrections_hgcalV9 = cms.vdouble(1.00, 1.00, 1.01, 1.01, 1.02, 1.01, 1.01, 1.01)

from Configuration.Eras.Modifier_phase2_hgcalV9_cff import phase2_hgcalV9
phase2_hgcalV9.toModify(hadronCorrections, value = hadronCorrections_hgcalV9)
phase2_hgcalV9.toModify(egammaCorrections, value = egammaCorrections_hgcalV9)
