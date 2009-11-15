# TODO: change all names consisting of 'allLayer0Taus' for less misleading ones
import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.RecoPFTauTag_cff import *
from RecoTauTag.TauTagTools.PFTauSelector_cfi  import pfTauSelector

''' 

pfTaus_cff

Specify the prototype/default configuration of 'layer0Taus', which is a selected
collection of taus that is used as an input to the layer1Taus.  The layer0 is
constructed by:
    * Rerunning a tau algorithm (fixedConePFTaus, shrinkingConePFTaus, etc)
    * Cloning and running a set of discriminants for this algorithm so they are
      independent of other cfis
    * Constructing layer0Taus via a PFTauSelector using the cloned discriminants
    * In PhysicsTools.PatAlgos.tools.pfTools the regular discriminants are
      modified to take the allLayer0Taus as input.  The original discriminant
      labels are kept (i.e. fixedConePFTauDiscriminationByIsolation) but the Tau
      source is defined as allLayer0Taus

'''

# The isolation discriminator requires this as prediscriminant, 
# as all sensical taus must have at least one track
allLayer0TausDiscriminationByLeadingTrackFinding = \
    shrinkingConePFTauDiscriminationByLeadingTrackFinding.clone()

# The actual selections on layer0Taus
allLayer0TausDiscriminationByLeadingPionPtCut = \
    shrinkingConePFTauDiscriminationByLeadingPionPtCut.clone()

allLayer0TausDiscriminationByIsolation = \
    shrinkingConePFTauDiscriminationByIsolation.clone()
allLayer0TausDiscriminationByIsolation.Prediscriminants.leadTrack.Producer = \
    "allLayer0TausDiscriminationByLeadingTrackFinding"

# Sequence to reproduce taus and compute our cloned discriminants
allLayer0TausBaseSequence = cms.Sequence(
    shrinkingConePFTauProducer +
    allLayer0TausDiscriminationByLeadingTrackFinding +
    allLayer0TausDiscriminationByLeadingPionPtCut +
    allLayer0TausDiscriminationByIsolation
    )

ic5PFJetTracksAssociatorAtVertex.jets = 'pfJets'

# Select taus from given collection that pass cloned discriminants
allLayer0Taus = pfTauSelector.clone()
allLayer0Taus.src = cms.InputTag("shrinkingConePFTauProducer")
allLayer0Taus.discriminators = cms.VPSet(
    cms.PSet( discriminator=cms.InputTag("allLayer0TausDiscriminationByLeadingPionPtCut"),selectionCut=cms.double(0.5) ),
    cms.PSet( discriminator=cms.InputTag("allLayer0TausDiscriminationByIsolation"),selectionCut=cms.double(0.5) )
    )

pfRecoTauTagInfoProducer.PFCandidateProducer = 'pfNoElectron'

pfTauSequence = cms.Sequence(
    ic5PFJetTracksAssociatorAtVertex + 
    pfRecoTauTagInfoProducer + 
    allLayer0TausBaseSequence + 
    allLayer0Taus 
    )


