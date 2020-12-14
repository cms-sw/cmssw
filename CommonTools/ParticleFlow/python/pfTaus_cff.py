import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.RecoPFTauTag_cff import *
from RecoTauTag.RecoTau.pfTauSelector_cfi  import pfTauSelector
import RecoTauTag.RecoTau.RecoTauCleanerPlugins as cleaners
import RecoJets.JetProducers.ak4PFJets_cfi as jetConfig

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




# PiZeroProducers

pfJetsLegacyHPSPiZeros = ak4PFJetsLegacyHPSPiZeros.clone(
    jetSrc = "ak4PFJets"
)
pfTauPFJetsRecoTauChargedHadrons = ak4PFJetsRecoTauChargedHadrons.clone()

pfTauTagInfoProducer = pfRecoTauTagInfoProducer.clone(
    PFCandidateProducer = jetConfig.ak4PFJets.src ,
    PFJetTracksAssociatorProducer = 'pfJetTracksAssociatorAtVertex'
)
# Clone tau producer
pfTausProducer = hpsPFTauProducer.clone(
    src = "pfTausProducerSansRefs"
)
pfTausCombiner = combinatoricRecoTaus.clone(
    jetSrc           = "ak4PFJets",
    piZeroSrc        = "pfJetsLegacyHPSPiZeros",
    chargedHadronSrc = "pfTauPFJetsRecoTauChargedHadrons"
)
for mod in pfTausCombiner.modifiers:
    if mod.name == "TTIworkaround": mod.pfTauTagInfoSrc = "pfTauTagInfoProducer"

pfTausSelectionDiscriminator = hpsSelectionDiscriminator.clone(
    PFTauProducer = "pfTausCombiner"
)
pfTausProducerSansRefs = hpsPFTauProducerSansRefs.clone(
    src = "pfTausCombiner",
    outputSelection = "",
    verbosity = 0,
    cleaners = [
        cleaners.unitCharge,
        cms.PSet(
            name = cms.string("leadStripPtLt2_5"),
            plugin = cms.string("RecoTauStringCleanerPlugin"),
            tolerance = cleaners.tolerance_default,
            selection = cms.string("signalPiZeroCandidates().size() = 0 | signalPiZeroCandidates()[0].pt > 2.5"),
            selectionPassFunction = cms.string("0"),
            selectionFailValue = cms.double(1e3)
        ),
        cms.PSet(
            name = cms.string("HPS_Select"),
            plugin = cms.string("RecoTauDiscriminantCleanerPlugin"),
            tolerance = cleaners.tolerance_default,
            src = cms.InputTag("pfTausSelectionDiscriminator"),
        ),
        cleaners.combinedIsolation
    ]
)

#cloning discriminants

pfTausDiscriminationByDecayModeFinding = hpsPFTauDiscriminationByDecayModeFinding.clone(
    PFTauProducer = "pfTausProducer"
)

pfTausrequireDecayMode = cms.PSet(
    BooleanOperator = cms.string("and"),
    decayMode = cms.PSet(
	Producer = cms.InputTag('pfTausDiscriminationByDecayModeFinding'),
	cut = cms.double(0.5)
    )
)

pfTausDiscriminationByIsolation= hpsPFTauBasicDiscriminators.clone(
    PFTauProducer = "pfTausProducer",
    Prediscriminants = pfTausrequireDecayMode.clone()
)

# Sequence to reproduce taus and compute our cloned discriminants
pfTausBaseSequence = cms.Sequence(
   pfJetsLegacyHPSPiZeros +
   pfTauPFJetsRecoTauChargedHadrons +
   pfTausCombiner +
   pfTausSelectionDiscriminator +
   pfTausProducerSansRefs +
   pfTausProducer +
   pfTausDiscriminationByDecayModeFinding *
   pfTausDiscriminationByIsolation
)

# Associate track to pfJets
pfJetTracksAssociatorAtVertex = ak4PFJetTracksAssociatorAtVertex.clone(
    jets = "ak4PFJets"
)

pfTauPileUpVertices = cms.EDFilter(
    "RecoTauPileUpVertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    minTrackSumPt = cms.double(5),
    filter = cms.bool(False),
)


pfTauTagInfoProducer = pfRecoTauTagInfoProducer.clone(
    PFCandidateProducer = jetConfig.ak4PFJets.src ,
    PFJetTracksAssociatorProducer = 'pfJetTracksAssociatorAtVertex'
)
pfTausPreSequence = cms.Sequence(
    pfJetTracksAssociatorAtVertex +
    recoTauAK4PFJets08Region +
    pfTauPileUpVertices +
    pfTauTagInfoProducer
)

# Select taus from given collection that pass cloned discriminants
pfTaus = pfTauSelector.clone(
    src = "pfTausProducer",
    discriminators = cms.VPSet(
        cms.PSet( 
	    discriminator=cms.InputTag("pfTausDiscriminationByDecayModeFinding"),
            selectionCut=cms.double(0.5) 
	),
    )
)
pfTausPtrs = cms.EDProducer("PFTauFwdPtrProducer",
                            src=cms.InputTag("pfTaus")
)

pfTauSequence = cms.Sequence(
    pfTausPreSequence +
    pfTausBaseSequence +
    pfTaus +
    pfTausPtrs
)
