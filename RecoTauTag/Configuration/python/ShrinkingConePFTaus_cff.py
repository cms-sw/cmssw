import FWCore.ParameterSet.Config as cms
import copy

"""
        Defines producers and discriminants for the "shrinkingCone" PFTau

        The sequence provided @ the end of the file,

                produceAndDiscriminateShrinkingConePFTaus

        produces the shrinking cone PFTau and all its associated discriminants

        Signal/Iso cone parameters:
           SignalCone for tracks           - 5/ET in DR from lead object, min 0.07, max 0.15
           SignalCone for ECAL/HCAL        - 0.15 in DR from lead object
           Isolation cone (all types0      - 0.50 in DR from lead object
"""

# Backwards compatability
# Get the decay mode reconstruction producer
from RecoTauTag.RecoTau.PFRecoTauDecayModeDeterminator_cfi                          import *
shrinkingConePFTauDecayModeProducer               = copy.deepcopy(pfTauDecayMode)
shrinkingConePFTauDecayModeProducer.PFTauProducer = 'shrinkingConePFTauProducer'

# Store the reco'd decay modes in a simple container
from RecoTauTag.RecoTau.PFRecoTauDecayModeIndexProducer_cfi                             import *
shrinkingConePFTauDecayModeIndexProducer                        = copy.deepcopy(pfTauDecayModeIndexProducer)
shrinkingConePFTauDecayModeIndexProducer.PFTauProducer          = cms.InputTag("shrinkingConePFTauProducer")
shrinkingConePFTauDecayModeIndexProducer.PFTauDecayModeProducer = cms.InputTag("shrinkingConePFTauDecayModeProducer")
# End backwards compatability

from RecoTauTag.TauTagTools.TauNeuralClassifiers_cfi import *

from RecoTauTag.RecoTau.RecoTauShrinkingConeProducer_cfi import \
        shrinkingConeRecoTaus as shrinkingConePFTauProducerSansRefs

shrinkingConePFTauProducer = cms.EDProducer(
    "RecoTauPiZeroUnembedder",
    src = cms.InputTag("shrinkingConePFTauProducerSansRefs")
)
shrinkingConePFTauProducer.builders = shrinkingConePFTauProducerSansRefs.builders
shrinkingConePFTauProducer.modifiers = shrinkingConePFTauProducerSansRefs.modifiers

# Define the discriminators for this tau
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackFinding_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackPtCut_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByTrackIsolation_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByECALIsolation_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectron_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuon_cfi import *
#Discriminators using leading Pion
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolationUsingLeadingPion_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingPionPtCut_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByTrackIsolationUsingLeadingPion_cfi import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByECALIsolationUsingLeadingPion_cfi import *

# Load helper functions to change the source of the discriminants
from RecoTauTag.RecoTau.TauDiscriminatorTools import *

#copying Discriminator ByLeadingTrack(finding and pt_cut)
shrinkingConePFTauDiscriminationByLeadingTrackFinding = \
        copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackFinding)
setTauSource(shrinkingConePFTauDiscriminationByLeadingTrackFinding,
             'shrinkingConePFTauProducer')

shrinkingConePFTauDiscriminationByLeadingTrackPtCut = \
        copy.deepcopy(pfRecoTauDiscriminationByLeadingTrackPtCut)
setTauSource(shrinkingConePFTauDiscriminationByLeadingTrackPtCut,
             'shrinkingConePFTauProducer')

#copying Discriminator ByPionTrackPtCut
shrinkingConePFTauDiscriminationByLeadingPionPtCut = \
        copy.deepcopy(pfRecoTauDiscriminationByLeadingPionPtCut)
setTauSource(shrinkingConePFTauDiscriminationByLeadingPionPtCut,
             'shrinkingConePFTauProducer')

#copying the Discriminator by Isolation
shrinkingConePFTauDiscriminationByIsolation = \
        copy.deepcopy(pfRecoTauDiscriminationByIsolation)
setTauSource(shrinkingConePFTauDiscriminationByIsolation,
             'shrinkingConePFTauProducer')

shrinkingConePFTauDiscriminationByTrackIsolation = \
        copy.deepcopy(pfRecoTauDiscriminationByTrackIsolation)
setTauSource(shrinkingConePFTauDiscriminationByTrackIsolation,
             'shrinkingConePFTauProducer')

shrinkingConePFTauDiscriminationByECALIsolation = \
        copy.deepcopy(pfRecoTauDiscriminationByECALIsolation)
setTauSource(shrinkingConePFTauDiscriminationByECALIsolation,
             'shrinkingConePFTauProducer')

#copying the Discriminator by Isolation for leadingPion
shrinkingConePFTauDiscriminationByIsolationUsingLeadingPion = \
        copy.deepcopy(pfRecoTauDiscriminationByIsolationUsingLeadingPion)
setTauSource(shrinkingConePFTauDiscriminationByIsolationUsingLeadingPion,
             'shrinkingConePFTauProducer')

shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion = \
        copy.deepcopy(pfRecoTauDiscriminationByTrackIsolationUsingLeadingPion)
setTauSource(shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion,
             'shrinkingConePFTauProducer')

shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion = \
        copy.deepcopy(pfRecoTauDiscriminationByECALIsolationUsingLeadingPion)
setTauSource(shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion,
             'shrinkingConePFTauProducer')

#copying discriminator against electrons and muons
shrinkingConePFTauDiscriminationAgainstElectron = \
        copy.deepcopy(pfRecoTauDiscriminationAgainstElectron)
setTauSource(shrinkingConePFTauDiscriminationAgainstElectron,
             'shrinkingConePFTauProducer')

shrinkingConePFTauDiscriminationAgainstMuon = \
        copy.deepcopy(pfRecoTauDiscriminationAgainstMuon)
setTauSource(shrinkingConePFTauDiscriminationAgainstMuon,
             'shrinkingConePFTauProducer')

produceAndDiscriminateShrinkingConePFTaus = cms.Sequence(
      shrinkingConePFTauProducerSansRefs*
      shrinkingConePFTauProducer*
      shrinkingConePFTauDiscriminationByLeadingTrackFinding*
      shrinkingConePFTauDiscriminationByLeadingTrackPtCut*
      shrinkingConePFTauDiscriminationByLeadingPionPtCut*
      shrinkingConePFTauDiscriminationByIsolation*
      shrinkingConePFTauDiscriminationByTrackIsolation*
      shrinkingConePFTauDiscriminationByECALIsolation*
      shrinkingConePFTauDiscriminationByIsolationUsingLeadingPion*
      shrinkingConePFTauDiscriminationByTrackIsolationUsingLeadingPion*
      shrinkingConePFTauDiscriminationByECALIsolationUsingLeadingPion*
      shrinkingConePFTauDiscriminationAgainstElectron*
      shrinkingConePFTauDiscriminationAgainstMuon
      )

produceShrinkingConeDiscriminationByTauNeuralClassifier = cms.Sequence(
      shrinkingConePFTauDiscriminationByTaNC*
      shrinkingConePFTauDiscriminationByTaNCfrOnePercent*
      shrinkingConePFTauDiscriminationByTaNCfrHalfPercent*
      shrinkingConePFTauDiscriminationByTaNCfrQuarterPercent*
      shrinkingConePFTauDiscriminationByTaNCfrTenthPercent
)


