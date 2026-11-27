"""pat::GenJet chain.

Input: the generator jets MiniAOD already uses, made in the same job:
  ak4GenJetsNoNu, ak8GenJetsNoNu, ak8GenJetsNoNuSoftDrop (+ SubJets), packedGenParticles (+ its association to genParticles).
None of them is defined here, nothing existing is modified.

Names: every helper module is called <reco label>ForPatGenJets; the producers strip that postfix (labelPostfixesToStrip), so the
user-float labels equal the reco ones after stripping, e.g. NjettinessAK8Puppi:tau1, nb1AK8PuppiSoftDrop:ecfN2.

The whole chain runs in the same job as the generator jets; patGenJetsTask collects its modules.
"""
import FWCore.ParameterSet.Config as cms

from PhysicsTools.PatAlgos.patGenJetProducer_cfi import patGenJetProducer as _patGenJetProducer
from PhysicsTools.PatAlgos.patGenJetSelector_cfi import patGenJetSelector as _patGenJetSelector
from PhysicsTools.PatAlgos.patGenJetSubstructurePacker_cfi import patGenJetSubstructurePacker as _patGenJetSubstructurePacker

# prototypes are imported under underscore names: process.load adds every public cms object of this file to the process
from RecoJets.JetProducers.nJettinessAdder_cfi import Njettiness as _Njettiness
from RecoJets.JetProducers.ECF_cff import ecfNbeta1 as _ecfNbeta1, ecfNbeta2 as _ecfNbeta2

from PhysicsTools.PatAlgos.tools.patGenJetHelpers import STRIP, userDataPSet

# helpers (userDataPSet) are in patGenJetHelpers.py

# ---------------------------------------------------------------------------------------------------------
# substructure modules: weights off (gen jets are unweighted)

NjettinessAK8PuppiForPatGenJets = _Njettiness.clone(src='ak8GenJetsNoNu', srcWeights='')
nb1AK8PuppiSoftDropForPatGenJets = _ecfNbeta1.clone(src='ak8GenJetsNoNuSoftDrop', srcWeights='', cuts=cms.vstring('', '', 'pt > 250'))
nb2AK8PuppiSoftDropForPatGenJets = _ecfNbeta2.clone(src='ak8GenJetsNoNuSoftDrop', srcWeights='', cuts=cms.vstring('', '', 'pt > 250'))
NjettinessAK8SubjetsForPatGenJets = _Njettiness.clone(src=cms.InputTag('ak8GenJetsNoNuSoftDrop', 'SubJets'), srcWeights='')
nb1AK8PuppiSoftDropSubjetsForPatGenJets = _ecfNbeta1.clone(src=cms.InputTag('ak8GenJetsNoNuSoftDrop', 'SubJets'), srcWeights='')
nb2AK8PuppiSoftDropSubjetsForPatGenJets = _ecfNbeta2.clone(src=cms.InputTag('ak8GenJetsNoNuSoftDrop', 'SubJets'), srcWeights='')

ak8PFJetsPuppiSoftDropMassForPatGenJets = cms.EDProducer("RecoJetDeltaRValueMapProducer",
    src=cms.InputTag('ak8GenJetsNoNu'),
    matched=cms.InputTag('ak8GenJetsNoNuSoftDrop'),
    distMax=cms.double(0.8),
    value=cms.string('mass'),
)

# ---------------------------------------------------------------------------------------------------------
# pat::GenJet producers

# AK4: no generator-side user data
patGenJetsAK4 = _patGenJetProducer.clone(
    jetSource='ak4GenJetsNoNu',
    packedGenParticles='packedGenParticles',
    userData=userDataPSet())

# AK8 fat jets: N-subjettiness 1..4 and soft-drop mass as in applySubstructure_cff.py:126-133
patGenJetsAK8 = _patGenJetProducer.clone(
    jetSource='ak8GenJetsNoNu',
    packedGenParticles='packedGenParticles',
    userData=userDataPSet(floats=[
        cms.InputTag('NjettinessAK8PuppiForPatGenJets', 'tau1'),
        cms.InputTag('NjettinessAK8PuppiForPatGenJets', 'tau2'),
        cms.InputTag('NjettinessAK8PuppiForPatGenJets', 'tau3'),
        cms.InputTag('NjettinessAK8PuppiForPatGenJets', 'tau4'),
        cms.InputTag('ak8PFJetsPuppiSoftDropMassForPatGenJets'),
    ]))

# slimmedGenJetsAK8 selection (slimmedGenJets_cfi.py:16-18)
selectedPatGenJetsAK8 = _patGenJetSelector.clone(
    src='patGenJetsAK8',
    cut='pt > 80',
    cutLoose='pt > 10.',
    nLoose=3)

# soft-drop subjets: N-subjettiness and ECFs as in applySubstructure_cff.py:90-92, :94-96
patGenJetsAK8SoftDropSubjets = _patGenJetProducer.clone(
    jetSource=cms.InputTag('ak8GenJetsNoNuSoftDrop', 'SubJets'),
    packedGenParticles='packedGenParticles',
    userData=userDataPSet(floats=[
        cms.InputTag('nb1AK8PuppiSoftDropSubjetsForPatGenJets', 'ecfN2'),
        cms.InputTag('nb1AK8PuppiSoftDropSubjetsForPatGenJets', 'ecfN3'),
        cms.InputTag('nb2AK8PuppiSoftDropSubjetsForPatGenJets', 'ecfN2'),
        cms.InputTag('nb2AK8PuppiSoftDropSubjetsForPatGenJets', 'ecfN3'),
        cms.InputTag('NjettinessAK8SubjetsForPatGenJets', 'tau1'),
        cms.InputTag('NjettinessAK8SubjetsForPatGenJets', 'tau2'),
        cms.InputTag('NjettinessAK8SubjetsForPatGenJets', 'tau3'),
        cms.InputTag('NjettinessAK8SubjetsForPatGenJets', 'tau4'),
    ]))

# one module for BoostedJetMerger + JetSubstructurePacker; label SoftDropPuppi as reco
packedPatGenJetsAK8 = _patGenJetSubstructurePacker.clone(
    jetSrc='selectedPatGenJetsAK8',
    distMax=0.8,
    fixDaughters=False,
    labelPostfixesToStrip=[STRIP],
    algos=cms.VPSet(cms.PSet(
        label=cms.string('SoftDropPuppi'),
        groomedJets=cms.InputTag('ak8GenJetsNoNuSoftDrop'),
        subjets=cms.InputTag('patGenJetsAK8SoftDropSubjets'),
        groomedUserFloats=cms.VInputTag(
            cms.InputTag('nb1AK8PuppiSoftDropForPatGenJets', 'ecfN2'),
            cms.InputTag('nb1AK8PuppiSoftDropForPatGenJets', 'ecfN3'),
            cms.InputTag('nb2AK8PuppiSoftDropForPatGenJets', 'ecfN2'),
            cms.InputTag('nb2AK8PuppiSoftDropForPatGenJets', 'ecfN3')),
        subjetInstance=cms.string('SubJets'))))

patGenJetsTask = cms.Task(
    NjettinessAK8PuppiForPatGenJets,
    nb1AK8PuppiSoftDropForPatGenJets,
    nb2AK8PuppiSoftDropForPatGenJets,
    NjettinessAK8SubjetsForPatGenJets,
    nb1AK8PuppiSoftDropSubjetsForPatGenJets,
    nb2AK8PuppiSoftDropSubjetsForPatGenJets,
    ak8PFJetsPuppiSoftDropMassForPatGenJets,
    patGenJetsAK4,
    patGenJetsAK8,
    selectedPatGenJetsAK8,
    patGenJetsAK8SoftDropSubjets,
    packedPatGenJetsAK8,
)
