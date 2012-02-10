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

# Clone tau producer
pfTausProducerSansRefs = shrinkingConePFTauProducerSansRefs.clone()
pfTausProducer = shrinkingConePFTauProducer.clone()
pfTausProducer.src = "pfTausProducerSansRefs"

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
    pfTausProducerSansRefs +
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
pfTausProducerSansRefs.jetSrc = pfJetTracksAssociatorAtVertex.jets
pfTausProducer.jetSrc = pfTausProducerSansRefs.jetSrc
# is it correct collection w/o good leptons
pfTausProducerSansRefs.builders[0].pfCandSrc = pfJets.src
pfTausProducer.builders[0].pfCandSrc = pfTausProducerSansRefs.builders[0].pfCandSrc

pfTauPileUpVertices = cms.EDFilter(
    "RecoTauPileUpVertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    minTrackSumPt = cms.double(5),
    filter = cms.bool(False),
)

# PiZeroProducers
pfJetsPiZeros = ak5PFJetsRecoTauPiZeros.clone()
pfJetsLegacyTaNCPiZeros = ak5PFJetsLegacyTaNCPiZeros.clone()
pfJetsLegacyHPSPiZeros = ak5PFJetsLegacyHPSPiZeros.clone()

pfJetsPiZeros.jetSrc = pfJetTracksAssociatorAtVertex.jets
pfJetsLegacyTaNCPiZeros.jetSrc = pfJetTracksAssociatorAtVertex.jets
pfJetsLegacyHPSPiZeros.jetSrc = pfJetTracksAssociatorAtVertex.jets

pfTauPFJets08Region = recoTauAK5PFJets08Region.clone()
pfTauPFJets08Region.src = pfJetTracksAssociatorAtVertex.jets
pfTauPFJets08Region.pfSrc = pfJets.src
pfJetsPiZeros.jetRegionSrc = 'pfTauPFJets08Region'
pfJetsLegacyTaNCPiZeros.jetRegionSrc = 'pfTauPFJets08Region'
pfJetsLegacyHPSPiZeros.jetRegionSrc = 'pfTauPFJets08Region'
pfTausProducerSansRefs.piZeroSrc = "pfJetsLegacyTaNCPiZeros"
pfTausProducerSansRefs.jetRegionSrc = pfJetsLegacyTaNCPiZeros.jetRegionSrc
pfTausProducer.piZeroSrc = pfTausProducerSansRefs.piZeroSrc
pfTausProducer.jetRegionSrc = pfTausProducerSansRefs.jetRegionSrc

pfTauTagInfoProducer = pfRecoTauTagInfoProducer.clone()
pfTauTagInfoProducer.PFCandidateProducer = pfJets.src
pfTauTagInfoProducer.PFJetTracksAssociatorProducer = 'pfJetTracksAssociatorAtVertex'

pfTausProducerSansRefs.modifiers[1] = cms.PSet(
    name = cms.string("pfTauTTIworkaround"),
    plugin = cms.string("RecoTauTagInfoWorkaroundModifer"),
    pfTauTagInfoSrc = cms.InputTag("pfTauTagInfoProducer"),
)
pfTausProducer.modifiers[1] = pfTausProducerSansRefs.modifiers[1]

pfTausPreSequence = cms.Sequence(
    pfJetTracksAssociatorAtVertex + 
    pfTauPFJets08Region +
    pfTauPileUpVertices +
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


