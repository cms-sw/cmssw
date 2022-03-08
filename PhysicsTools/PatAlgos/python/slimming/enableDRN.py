import FWCore.ParameterSet.Config as cms
from RecoEgamma.EgammaTools.egammaObjectModificationsInMiniAOD_cff import egamma_modifications

def customise(process):
  process.load("PhysicsTools.PatAlgos.slimming.patPhotonDRNCorrector_cfi")
  process.DRNTask = cms.Task(process.patPhotonsDRN)
  process.schedule.associate(process.DRNTask)
  egamma_modifications.append(
    cms.PSet( modifierName = cms.string("EGRegressionModifierDRN"),
      patPhotons = cms.PSet(
        source = cms.InputTag('selectedPatPhotons'),
        correctionsSource = cms.InputTag('patPhotonsDRN')
      )
    )
  )

  return process
