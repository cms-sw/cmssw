import FWCore.ParameterSet.Config as cms

from EgammaAnalysis.EgammaIsolationProducers.egammaElectronTkIsolation_cfi import *
from EgammaAnalysis.EgammaIsolationProducers.egammaElectronTkNumIsolation_cfi import *
from EgammaAnalysis.EgammaIsolationProducers.egammaHOETower_cfi import *
from EgammaAnalysis.EgammaIsolationProducers.egammaEcalRelIsolationSequence_cff import *
#NO: now we run ISO on AOD first
#  replace egammaHOETower.emObjectProducer = allLayer0Electrons
#  replace egammaElectronTkIsolation.electronProducer = allLayer0Electrons
#  replace egammaElectronTkNumIsolation.electronProducer = allLayer0Electrons
#  replace egammaEcalRelIsolation.emObjectProducer = allLayer0Electrons
# read and convert to ValueMap<float>
patAODElectronIsolations = cms.EDFilter("MultipleNumbersToValueMaps",
    associations = cms.VInputTag(cms.InputTag("egammaHOETower"), cms.InputTag("egammaElectronTkIsolation"), cms.InputTag("egammaElectronTkNumIsolation"), cms.InputTag("egammaEcalRelIsolation")),
    collection = cms.InputTag("pixelMatchGsfElectrons")
)

layer0EgammaHOETower = cms.EDFilter("CandValueMapSkimmerFloat",
    association = cms.InputTag("patAODElectronIsolations","egammaHOETower"),
    collection = cms.InputTag("allLayer0Electrons"),
    backrefs = cms.InputTag("allLayer0Electrons")
)

layer0EgammaElectronTkIsolation = cms.EDFilter("CandValueMapSkimmerFloat",
    association = cms.InputTag("patAODElectronIsolations","egammaElectronTkIsolation"),
    collection = cms.InputTag("allLayer0Electrons"),
    backrefs = cms.InputTag("allLayer0Electrons")
)

layer0EgammaElectronTkNumIsolation = cms.EDFilter("CandValueMapSkimmerFloat",
    association = cms.InputTag("patAODElectronIsolations","egammaElectronTkNumIsolation"),
    collection = cms.InputTag("allLayer0Electrons"),
    backrefs = cms.InputTag("allLayer0Electrons")
)

layer0EgammaEcalRelIsolation = cms.EDFilter("CandValueMapSkimmerFloat",
    association = cms.InputTag("patAODElectronIsolations","egammaEcalRelIsolation"),
    collection = cms.InputTag("allLayer0Electrons"),
    backrefs = cms.InputTag("allLayer0Electrons")
)

patAODElectronIsolation = cms.Sequence(egammaHOETower+egammaElectronTkIsolation+egammaElectronTkNumIsolation+egammaEcalRelIsolationSequence+patAODElectronIsolations)
patLayer0ElectronIsolation = cms.Sequence(layer0EgammaHOETower*layer0EgammaElectronTkIsolation*layer0EgammaElectronTkNumIsolation*layer0EgammaEcalRelIsolation)

