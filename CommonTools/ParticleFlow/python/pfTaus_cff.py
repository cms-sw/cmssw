import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.RecoPFTauTag_cff import *
from RecoTauTag.TauTagTools.PFTauSelector_cfi  import pfTauSelector

from CommonTools.ParticleFlow.pfJets_cff import pfJets

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

# The isolation discriminator requires this as prediscriminant, 
# as all sensical taus must have at least one track
pfTausDiscriminationByLeadingTrackFinding = \
    shrinkingConePFTauDiscriminationByLeadingTrackFinding.clone()

# The actual selections on pfTaus
pfTausDiscriminationByLeadingPionPtCut = \
    shrinkingConePFTauDiscriminationByLeadingPionPtCut.clone()

pfTausDiscriminationByIsolation = \
    shrinkingConePFTauDiscriminationByIsolation.clone()
pfTausDiscriminationByIsolation.Prediscriminants.leadTrack.Producer = \
    "pfTausDiscriminationByLeadingTrackFinding"

# Sequence to reproduce taus and compute our cloned discriminants
pfTausBaseSequence = cms.Sequence(
    shrinkingConePFTauProducer +
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
shrinkingConePFTauProducer.jetSrc = pfJetTracksAssociatorAtVertex.jets
# is it correct collection w/o good leptons
shrinkingConePFTauProducer.builders[0].pfCandSrc = pfJets.src

# PiZeroProducers
pfJetsLegacyTaNCPiZeros = ak5PFJetsLegacyTaNCPiZeros.clone()
pfJetsLegacyTaNCPiZeros.jetSrc = pfJetTracksAssociatorAtVertex.jets
pfTauPFJets08Region = recoTauAK5PFJets08Region.clone()
pfTauPFJets08Region.src = pfJetTracksAssociatorAtVertex.jets
pfTauPFJets08Region.pfSrc = pfJets.src
pfJetsLegacyTaNCPiZeros.jetRegionSrc = 'pfTauPFJets08Region'
shrinkingConePFTauProducer.piZeroSrc = "pfJetsLegacyTaNCPiZeros"
shrinkingConePFTauProducer.jetRegionSrc = pfJetsLegacyTaNCPiZeros.jetRegionSrc

# Select taus from given collection that pass cloned discriminants
pfTaus = pfTauSelector.clone()
pfTaus.src = cms.InputTag("shrinkingConePFTauProducer")
pfTaus.discriminators = cms.VPSet(
    cms.PSet( discriminator=cms.InputTag("pfTausDiscriminationByLeadingPionPtCut"),selectionCut=cms.double(0.5) ),
    cms.PSet( discriminator=cms.InputTag("pfTausDiscriminationByIsolation"),selectionCut=cms.double(0.5) )
    )

pfRecoTauTagInfoProducer.PFCandidateProducer = pfJets.src
pfRecoTauTagInfoProducer.PFJetTracksAssociatorProducer = 'pfJetTracksAssociatorAtVertex'

pfTauSequence = cms.Sequence(
    pfJetTracksAssociatorAtVertex + 
    pfRecoTauTagInfoProducer + 
    pfTauPFJets08Region +
    pfJetsLegacyTaNCPiZeros +
    pfTausBaseSequence + 
    pfTaus 
    )


