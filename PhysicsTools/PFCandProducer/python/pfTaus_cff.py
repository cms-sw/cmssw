import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.RecoPFTauTag_cff import *
from RecoTauTag.TauTagTools.PFTauSelector_cfi  import pfTauSelector

from PhysicsTools.PFCandProducer.pfJets_cff import pfJets

''' 

pfTaus_cff

Specify the prototype/default configuration of 'pfTaus', which is a selected
collection of taus that is used as an input to the 'patTaus'. The pf2pat tau
selection is constructed by:
    * Rerunning a tau algorithm (fixedConePFTaus, shrinkingConePFTaus, etc)
    * Cloning and running a set of discriminants for this algorithm so they are
      independent of other cfis
    * Constructing pfTaus via a PFTauSelector using the cloned discriminants
    * In PhysicsTools.PatAlgos.tools.pfTools the regular discriminants are
      modified to take the pfTaus as input.  The original discriminant
      labels are kept (i.e. fixedConePFTauDiscriminationByIsolation) but the Tau
      source is defined as pfTaus

'''

# Clone tau producer
pfTausProducer = shrinkingConePFTauProducer.clone()

# The isolation discriminator requires this as prediscriminant, 
# as all sensical taus must have at least one track
pfTausDiscriminationByLeadingTrackFinding = \
    shrinkingConePFTauDiscriminationByLeadingTrackFinding.clone()
pfTausDiscriminationByLeadingTrackFinding.PFTauProducer = "pfTausProducer"

# The actual selections on pfTaus
pfTausDiscriminationByLeadingPionPtCut = \
    shrinkingConePFTauDiscriminationByLeadingPionPtCut.clone()
pfTausDiscriminationByLeadingPionPtCut.PFTauProducer = "pfTausProducer"

pfTausDiscriminationByIsolation = \
    shrinkingConePFTauDiscriminationByIsolation.clone()
pfTausDiscriminationByIsolation.Prediscriminants.leadTrack.Producer = \
    "pfTausDiscriminationByLeadingTrackFinding"
pfTausDiscriminationByIsolation.PFTauProducer = "pfTausProducer"

# Sequence to reproduce taus and compute our cloned discriminants
pfTausBaseSequence = cms.Sequence(
    pfTausProducer +
    pfTausDiscriminationByLeadingTrackFinding +
    pfTausDiscriminationByLeadingPionPtCut +
    pfTausDiscriminationByIsolation
    )

# Associate track to pfJets
from RecoJets.JetAssociationProducers.j2tParametersVX_cfi import *
pfJetTracksAssociatorAtVertex = cms.EDProducer(
    "JetTracksAssociatorAtVertex",
    j2tParametersVX,
    jets = cms.InputTag("pfJets")
    )
pfTausProducer.jetSrc = pfJetTracksAssociatorAtVertex.jets
# is it correct collection w/o good leptons
pfTausProducer.builders[0].pfCandSrc = pfJets.src

# PiZeroProducers
pfJetsPiZeros = ak5PFJetsRecoTauPiZeros.clone()
pfJetsLegacyTaNCPiZeros = ak5PFJetsLegacyTaNCPiZeros.clone()
pfJetsLegacyHPSPiZeros = ak5PFJetsLegacyHPSPiZeros.clone()

pfJetsPiZeros.src = pfJetTracksAssociatorAtVertex.jets
pfJetsLegacyTaNCPiZeros.src = pfJetTracksAssociatorAtVertex.jets
pfJetsLegacyHPSPiZeros.src = pfJetTracksAssociatorAtVertex.jets


pfTauTagInfoProducer = pfRecoTauTagInfoProducer.clone()
pfTauTagInfoProducer.PFCandidateProducer = pfJets.src
pfTauTagInfoProducer.PFJetTracksAssociatorProducer = 'pfJetTracksAssociatorAtVertex'

pfTausProducer.modifiers[1] = cms.PSet(
    name = cms.string("pfTauTTIworkaround"),
    plugin = cms.string("RecoTauTagInfoWorkaroundModifer"),
    pfTauTagInfoSrc = cms.InputTag("pfTauTagInfoProducer"),
)

pfTausPreSequence = cms.Sequence(
    pfJetTracksAssociatorAtVertex + 
    pfTauTagInfoProducer +
    pfJetsPiZeros +
    pfJetsLegacyTaNCPiZeros +
    pfJetsLegacyHPSPiZeros
)

# Select taus from given collection that pass cloned discriminants
pfTaus = pfTauSelector.clone()
pfTaus.src = cms.InputTag("pfTausProducer")
pfTaus.discriminators = cms.VPSet(
    cms.PSet( discriminator=cms.InputTag("pfTausDiscriminationByLeadingPionPtCut"),selectionCut=cms.double(0.5) ),
    cms.PSet( discriminator=cms.InputTag("pfTausDiscriminationByIsolation"),selectionCut=cms.double(0.5) )
    )

pfTauSequence = cms.Sequence(
    pfTausPreSequence +
    pfTausBaseSequence + 
    pfTaus 
    )


