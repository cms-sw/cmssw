import FWCore.ParameterSet.Config as cms

from RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi import *
import RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi


# CutBased: Robust
cutbasedRobustElectron = RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi.eidCutBasedExt.clone()
cutbasedRobustElectron.electronQuality = cms.string('robust')

# CutBased: Loose
cutbasedLooseElectron = RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi.eidCutBasedExt.clone()
cutbasedLooseElectron.electronQuality = cms.string('loose')

# CutBased: Tight
cutbasedTightElectron = RecoEgamma.ElectronIdentification.electronIdCutBasedExt_cfi.eidCutBasedExt.clone()
cutbasedTightElectron.electronQuality = cms.string('tight')

# Likelihood
from RecoEgamma.ElectronIdentification.electronIdLikelihoodExt_cfi import *
lhElectron = RecoEgamma.ElectronIdentification.electronIdLikelihoodExt_cfi.eidLikelihoodExt.clone()

# Neural Net
from RecoEgamma.ElectronIdentification.electronIdNeuralNetExt_cfi import *
nnElectron = RecoEgamma.ElectronIdentification.electronIdNeuralNetExt_cfi.eidNeuralNetExt.clone()



# No Id
noIdElectronCands = cms.EDProducer("GsfElectronShallowCloneProducer",
    src = cms.InputTag("pixelMatchGsfElectrons")
)

noIdElectronMatch = cms.EDFilter("MCTruthDeltaRMatcher",
    pdgId = cms.vint32(11),
    src = cms.InputTag("noIdElectronCands"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticleCandidates")
)


cutbasedRobustElectronCands = cms.EDProducer("eidCandProducer",
    ElectronIDAssociationProducer = cms.string('cutbasedRobustElectron'),
    InputProducer = cms.string('pixelMatchGsfElectrons')
)

cutbasedRobustElectronMatch = cms.EDFilter("MCTruthDeltaRMatcher",
    pdgId = cms.vint32(11),
    src = cms.InputTag("cutbasedRobustElectronCands"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticleCandidates")
)

cutbasedLooseElectronCands = cms.EDProducer("eidCandProducer",
    ElectronIDAssociationProducer = cms.string('cutbasedLooseElectron'),
    InputProducer = cms.string('pixelMatchGsfElectrons')
)

cutbasedLooseElectronMatch = cms.EDFilter("MCTruthDeltaRMatcher",
    pdgId = cms.vint32(11),
    src = cms.InputTag("cutbasedLooseElectronCands"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticleCandidates")
)

cutbasedTightElectronCands = cms.EDProducer("eidCandProducer",
    ElectronIDAssociationProducer = cms.string('cutbasedTightElectron'),
    InputProducer = cms.string('pixelMatchGsfElectrons')
)

cutbasedTightElectronMatch = cms.EDFilter("MCTruthDeltaRMatcher",
    pdgId = cms.vint32(11),
    src = cms.InputTag("cutbasedTightElectronCands"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticleCandidates")
)

lhElectronCands = cms.EDProducer("eidCandProducer",
    ElectronIDAssociationProducer = cms.string('lhElectron'),
    InputProducer = cms.string('pixelMatchGsfElectrons')
)

lhElectronMatch = cms.EDFilter("MCTruthDeltaRMatcher",
    pdgId = cms.vint32(11),
    src = cms.InputTag("lhElectronCands"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticleCandidates")
)

nnElectronCands = cms.EDProducer("eidCandProducer",
    ElectronIDAssociationProducer = cms.string('nnElectron'),
    InputProducer = cms.string('pixelMatchGsfElectrons')
)

nnElectronMatch = cms.EDFilter("MCTruthDeltaRMatcher",
    pdgId = cms.vint32(11),
    src = cms.InputTag("nnElectronCands"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticleCandidates")
)

#
#
#
#
#
# Golden electron ##################
#
goldenElectron = cms.EDProducer("gsfCategoryProducer",
    GsfCategory = cms.string('golden'),
    InputProducer = cms.string('pixelMatchGsfElectrons'),
    isInCrack = cms.bool(False),
    isInEndCap = cms.bool(True),
    isInBarrel = cms.bool(True)
)

goldenElectronCands = cms.EDProducer("GsfElectronShallowCloneProducer",
    src = cms.InputTag("goldenElectron")
)

goldenElectronMatch = cms.EDFilter("MCTruthDeltaRMatcher",
    pdgId = cms.vint32(11),
    src = cms.InputTag("goldenElectronCands"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticleCandidates")
)

#
# BigBrem electron ##################
#
bigbremElectron = cms.EDProducer("gsfCategoryProducer",
    GsfCategory = cms.string('bigbrem'),
    InputProducer = cms.string('pixelMatchGsfElectrons'),
    isInCrack = cms.bool(False),
    isInEndCap = cms.bool(True),
    isInBarrel = cms.bool(True)
)

bigbremElectronCands = cms.EDProducer("GsfElectronShallowCloneProducer",
    src = cms.InputTag("bigbremElectron")
)

bigbremElectronMatch = cms.EDFilter("MCTruthDeltaRMatcher",
    pdgId = cms.vint32(11),
    src = cms.InputTag("bigbremElectronCands"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticleCandidates")
)

#
# Narrow electron ##################
#
narrowElectron = cms.EDProducer("gsfCategoryProducer",
    GsfCategory = cms.string('narrow'),
    InputProducer = cms.string('pixelMatchGsfElectrons'),
    isInCrack = cms.bool(False),
    isInEndCap = cms.bool(True),
    isInBarrel = cms.bool(True)
)

narrowElectronCands = cms.EDProducer("GsfElectronShallowCloneProducer",
    src = cms.InputTag("narrowElectron")
)

narrowElectronMatch = cms.EDFilter("MCTruthDeltaRMatcher",
    pdgId = cms.vint32(11),
    src = cms.InputTag("narrowElectronCands"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticleCandidates")
)

#
# Showering electron ##################
#
showeringElectron = cms.EDProducer("gsfCategoryProducer",
    GsfCategory = cms.string('shower'),
    InputProducer = cms.string('pixelMatchGsfElectrons'),
    isInCrack = cms.bool(False),
    isInEndCap = cms.bool(True),
    isInBarrel = cms.bool(True)
)

showeringElectronCands = cms.EDProducer("GsfElectronShallowCloneProducer",
    src = cms.InputTag("showeringElectron")
)

showeringElectronMatch = cms.EDFilter("MCTruthDeltaRMatcher",
    pdgId = cms.vint32(11),
    src = cms.InputTag("showeringElectronCands"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticleCandidates")
)

# *******************************************************
#
#
#
#
electronIdSequence = cms.Sequence(noIdElectronCands+noIdElectronMatch+cutbasedRobustElectron+cutbasedRobustElectronCands+cutbasedRobustElectronMatch+cutbasedLooseElectron+cutbasedLooseElectronCands+cutbasedLooseElectronMatch+cutbasedTightElectron+cutbasedTightElectronCands+cutbasedTightElectronMatch+lhElectron+lhElectronCands+lhElectronMatch+nnElectron+nnElectronCands+nnElectronMatch)
#
#
#
#
electronClassificationSequence = cms.Sequence(goldenElectron+goldenElectronCands+goldenElectronMatch+bigbremElectron+bigbremElectronCands+bigbremElectronMatch+narrowElectron+narrowElectronCands+narrowElectronMatch+showeringElectron+showeringElectronCands+showeringElectronMatch)


