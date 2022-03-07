import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.helpers import getPatAlgosToolsTask

def _enableDRN(process):
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

from Configuration.Eras.Era_Run2_2018_cff import Run2_2018
enableDRN = cms.ProcessModifier(Run2_2018, _enableDRN)
