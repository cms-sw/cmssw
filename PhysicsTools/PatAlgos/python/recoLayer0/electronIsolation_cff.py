import FWCore.ParameterSet.Config as cms

from EgammaAnalysis.EgammaIsolationProducers.egammaElectronTkIsolation_cfi import *
from EgammaAnalysis.EgammaIsolationProducers.egammaElectronTkNumIsolation_cfi import *
from EgammaAnalysis.EgammaIsolationProducers.egammaHOETower_cfi import *
from EgammaAnalysis.EgammaIsolationProducers.egammaTowerIsolation_cfi import *
from EgammaAnalysis.EgammaIsolationProducers.egammaEcalRelIsolationSequence_cff import *
from EgammaAnalysis.EgammaIsolationProducers.egammaEcalIsolationSequence_cff import *
patAODElectronIsolationLabels = cms.PSet(
    associations = cms.VInputTag(cms.InputTag("egammaTowerIsolation"), cms.InputTag("egammaHOETower"), cms.InputTag("egammaElectronTkIsolation"), cms.InputTag("egammaElectronTkNumIsolation"), cms.InputTag("egammaEcalRelIsolation"), 
        cms.InputTag("egammaEcalIsolation"))
)
patAODElectronIsolations = cms.EDFilter("MultipleNumbersToValueMaps",
    patAODElectronIsolationLabels,
    collection = cms.InputTag("pixelMatchGsfElectrons")
)

layer0ElectronIsolations = cms.EDFilter("CandManyValueMapsSkimmerFloat",
    patAODElectronIsolationLabels,
    commonLabel = cms.InputTag("patAODElectronIsolations"),
    collection = cms.InputTag("allLayer0Electrons"),
    backrefs = cms.InputTag("allLayer0Electrons")
)

patAODElectronIsolation = cms.Sequence(egammaTowerIsolation+egammaHOETower+egammaElectronTkIsolation+egammaElectronTkNumIsolation+egammaEcalRelIsolationSequence+egammaEcalIsolation+patAODElectronIsolations)
patLayer0ElectronIsolation = cms.Sequence(layer0ElectronIsolations)

