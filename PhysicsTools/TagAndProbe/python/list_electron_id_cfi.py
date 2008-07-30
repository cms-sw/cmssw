import FWCore.ParameterSet.Config as cms

from EgammaAnalysis.ElectronIDProducers.electronId_cfi import *
import EgammaAnalysis.ElectronIDProducers.electronId_cfi
# PTDR: Loose
ptdrLooseElectron = EgammaAnalysis.ElectronIDProducers.electronId_cfi.electronId.clone()
import EgammaAnalysis.ElectronIDProducers.electronId_cfi
# PTDR: Medium
ptdrMediumElectron = EgammaAnalysis.ElectronIDProducers.electronId_cfi.electronId.clone()
import EgammaAnalysis.ElectronIDProducers.electronId_cfi
# PTDR: Tight
ptdrTightElectron = EgammaAnalysis.ElectronIDProducers.electronId_cfi.electronId.clone()
import EgammaAnalysis.ElectronIDProducers.electronId_cfi
# CutBased: Robust
cutbasedRobustElectron = EgammaAnalysis.ElectronIDProducers.electronId_cfi.electronId.clone()
import EgammaAnalysis.ElectronIDProducers.electronId_cfi
# CutBased: Loose
cutbasedLooseElectron = EgammaAnalysis.ElectronIDProducers.electronId_cfi.electronId.clone()
import EgammaAnalysis.ElectronIDProducers.electronId_cfi
# CutBased: Tight
cutbasedTightElectron = EgammaAnalysis.ElectronIDProducers.electronId_cfi.electronId.clone()
import EgammaAnalysis.ElectronIDProducers.electronId_cfi
# Likelihood
lhElectron = EgammaAnalysis.ElectronIDProducers.electronId_cfi.electronId.clone()
import EgammaAnalysis.ElectronIDProducers.electronId_cfi
# Neural Net
nnElectron = EgammaAnalysis.ElectronIDProducers.electronId_cfi.electronId.clone()
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

ptdrLooseElectronCands = cms.EDProducer("eidCandProducer",
    ElectronIDAssociationProducer = cms.string('ptdrLooseElectron'),
    InputProducer = cms.string('pixelMatchGsfElectrons')
)

ptdrLooseElectronMatch = cms.EDFilter("MCTruthDeltaRMatcher",
    pdgId = cms.vint32(11),
    src = cms.InputTag("ptdrLooseElectronCands"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticleCandidates")
)

ptdrMediumElectronCands = cms.EDProducer("eidCandProducer",
    ElectronIDAssociationProducer = cms.string('ptdrMediumElectron'),
    InputProducer = cms.string('pixelMatchGsfElectrons')
)

ptdrMediumElectronMatch = cms.EDFilter("MCTruthDeltaRMatcher",
    pdgId = cms.vint32(11),
    src = cms.InputTag("ptdrMediumElectronCands"),
    distMin = cms.double(0.25),
    matched = cms.InputTag("genParticleCandidates")
)

ptdrTightElectronCands = cms.EDProducer("eidCandProducer",
    ElectronIDAssociationProducer = cms.string('ptdrTightElectron'),
    InputProducer = cms.string('pixelMatchGsfElectrons')
)

ptdrTightElectronMatch = cms.EDFilter("MCTruthDeltaRMatcher",
    pdgId = cms.vint32(11),
    src = cms.InputTag("ptdrTightElectronCands"),
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
electronIdSequence = cms.Sequence(electronId+noIdElectronCands+noIdElectronMatch+cms.SequencePlaceholder("ptdrLooseElectronId")+ptdrLooseElectronCands+ptdrLooseElectronMatch+cms.SequencePlaceholder("ptdrMediumElectronId")+ptdrMediumElectronCands+ptdrMediumElectronMatch+cms.SequencePlaceholder("ptdrTightElectronId")+ptdrTightElectronCands+ptdrTightElectronMatch+cms.SequencePlaceholder("cutbasedRobustElectronId")+cutbasedRobustElectronCands+cutbasedRobustElectronMatch+cms.SequencePlaceholder("cutbasedLooseElectronId")+cutbasedLooseElectronCands+cutbasedLooseElectronMatch+cms.SequencePlaceholder("cutbasedTightElectronId")+cutbasedTightElectronCands+cutbasedTightElectronMatch+cms.SequencePlaceholder("lhElectronId")+lhElectronCands+lhElectronMatch+cms.SequencePlaceholder("nnElectronId")+nnElectronCands+nnElectronMatch)
#
#
#
#
electronClassificationSequence = cms.Sequence(goldenElectron+goldenElectronCands+goldenElectronMatch+bigbremElectron+bigbremElectronCands+bigbremElectronMatch+narrowElectron+narrowElectronCands+narrowElectronMatch+showeringElectron+showeringElectronCands+showeringElectronMatch)
ptdrLooseElectron.algo_psets.append(cms.PSet(
    electronQuality = cms.string('loose')
))
ptdrMediumElectron.algo_psets.append(cms.PSet(
    electronQuality = cms.string('medium')
))
ptdrTightElectron.algo_psets.append(cms.PSet(
    electronQuality = cms.string('tight')
))
cutbasedRobustElectron.doPtdrId = False
cutbasedRobustElectron.doCutBased = True
cutbasedRobustElectron.algo_psets.append(cms.PSet(
    electronQuality = cms.string('robust')
))
cutbasedLooseElectron.doPtdrId = False
cutbasedLooseElectron.doCutBased = True
cutbasedLooseElectron.algo_psets.append(cms.PSet(
    electronQuality = cms.string('loose')
))
cutbasedTightElectron.doPtdrId = False
cutbasedTightElectron.doCutBased = True
cutbasedTightElectron.algo_psets.append(cms.PSet(
    electronQuality = cms.string('tight')
))
lhElectron.doPtdrId = False
lhElectron.doLikelihood = True
nnElectron.doPtdrId = False
nnElectron.doNeuralNet = True

