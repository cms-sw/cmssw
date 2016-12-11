import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.slimming.modifiedElectrons_cfi import modifiedElectrons
from PhysicsTools.PatAlgos.slimming.modifiedPhotons_cfi import modifiedPhotons
from EgammaAnalysis.ElectronTools.regressionModifier_cfi import regressionModifier

regressionElectrons = modifiedElectrons.clone()
regressionPhotons = modifiedPhotons.clone()

regressionModifier.ecalrechitsEB = cms.InputTag("reducedEgamma:reducedEBRecHits")
regressionModifier.ecalrechitsEE = cms.InputTag("reducedEgamma:reducedEERecHits")
regressionModifier.useLocalFile  = cms.bool(False)

egamma_modifications = cms.VPSet( )
egamma_modifications.append( regressionModifier )

regressionElectrons.modifierConfig.modifications = egamma_modifications
regressionPhotons.modifierConfig.modifications   = egamma_modifications

regressionApplication = cms.Sequence( regressionElectrons * regressionPhotons )
