import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask

def enableDRN(process):
  process.load("PhysicsTools.PatAlgos.slimming.patPhotonDRNCorrector_cfi")
  process.DRNTask = cms.Task(process.patPhotonsDRN)
  process.schedule.associate(process.DRNTask)
  process.slimmedPhotons.modifierConfig.modifications.append(
    cms.PSet( modifierName = cms.string("EGRegressionModifierDRN"),
      patPhotons = cms.PSet(
        source = cms.InputTag('selectedPatPhotons'),
        correctionsSource = cms.InputTag('patPhotonsDRN')
      )
    )
  )

  return process
