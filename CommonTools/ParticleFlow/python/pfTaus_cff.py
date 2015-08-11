import FWCore.ParameterSet.Config as cms

from RecoTauTag.Configuration.RecoPFTauTag_cff import *
from RecoTauTag.TauTagTools.PFTauSelector_cfi  import pfTauSelector
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

pfJetsLegacyHPSPiZeros = ak4PFJetsLegacyHPSPiZeros.clone()

pfJetsLegacyHPSPiZeros.jetSrc = cms.InputTag("ak4PFJets")

pfTauPFJets08Region = recoTauAK4PFJets08Region.clone()
pfTauPFJets08Region.src = cms.InputTag("ak4PFJets")
pfTauPFJetsRecoTauChargedHadrons = ak4PFJetsRecoTauChargedHadrons.clone()
pfTauPFJets08Region.pfSrc = cms.InputTag("particleFlow")
pfTauPFJetsRecoTauChargedHadrons.jetRegionSrc = 'pfTauPFJets08Region'

pfTauTagInfoProducer = pfRecoTauTagInfoProducer.clone()
pfTauTagInfoProducer.PFCandidateProducer = jetConfig.ak4PFJets.src
pfTauTagInfoProducer.PFJetTracksAssociatorProducer = 'pfJetTracksAssociatorAtVertex'

# Clone tau producer
pfTausProducer = hpsPFTauProducer.clone()
pfTausCombiner = combinatoricRecoTaus.clone()
pfTausCombiner.jetSrc= cms.InputTag("ak4PFJets")
pfTausCombiner.piZeroSrc= "pfJetsLegacyHPSPiZeros"
pfTausCombiner.jetRegionSrc='pfTauPFJets08Region'
pfTausCombiner.chargedHadronSrc='pfTauPFJetsRecoTauChargedHadrons'
pfTausCombiner.modifiers[3].pfTauTagInfoSrc=cms.InputTag("pfTauTagInfoProducer")
pfTausSelectionDiscriminator = hpsSelectionDiscriminator.clone()
pfTausSelectionDiscriminator.PFTauProducer = cms.InputTag("pfTausCombiner")
pfTausProducerSansRefs = hpsPFTauProducerSansRefs.clone()
pfTausProducerSansRefs = cms.EDProducer(
    "RecoTauCleaner",
    src = cms.InputTag("pfTausCombiner"),
    cleaners = cms.VPSet(
    cleaners.unitCharge,
    cms.PSet(
    name = cms.string("leadStripPtLt2_5"),
    plugin = cms.string("RecoTauStringCleanerPlugin"),
    selection = cms.string("signalPiZeroCandidates().size() = 0 | signalPiZeroCandidates()[0].pt > 2.5"),
    selectionPassFunction = cms.string("0"),
    selectionFailValue = cms.double(1e3)
    ),
    cms.PSet(
    name = cms.string("HPS_Select"),
    plugin = cms.string("RecoTauDiscriminantCleanerPlugin"),
    src = cms.InputTag("pfTausSelectionDiscriminator"),
    ),
    cleaners.combinedIsolation
    )
)



pfTausProducerSansRefs.src=cms.InputTag("pfTausCombiner")
pfTausProducer.src = cms.InputTag("pfTausProducerSansRefs")

#cloning discriminants

pfTausDiscriminationByDecayModeFinding = hpsPFTauDiscriminationByDecayModeFinding.clone()
pfTausDiscriminationByDecayModeFinding.PFTauProducer="pfTausProducer"

pfTausDiscriminationByIsolation= hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits.clone()
pfTausDiscriminationByIsolation.PFTauProducer="pfTausProducer"

pfTausrequireDecayMode = cms.PSet(
    BooleanOperator = cms.string("and"),
    decayMode = cms.PSet(
    Producer = cms.InputTag('pfTausDiscriminationByDecayModeFinding'),
    cut = cms.double(0.5)
    )
)

pfTausDiscriminationByIsolation.Prediscriminants=pfTausrequireDecayMode.clone()

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
pfJetTracksAssociatorAtVertex = ak4PFJetTracksAssociatorAtVertex.clone()
pfJetTracksAssociatorAtVertex.jets= cms.InputTag("ak4PFJets")

pfTauPileUpVertices = cms.EDFilter(
    "RecoTauPileUpVertexSelector",
    src = cms.InputTag("offlinePrimaryVertices"),
    minTrackSumPt = cms.double(5),
    filter = cms.bool(False),
)


pfTauTagInfoProducer = pfRecoTauTagInfoProducer.clone()
pfTauTagInfoProducer.PFCandidateProducer = jetConfig.ak4PFJets.src
pfTauTagInfoProducer.PFJetTracksAssociatorProducer = 'pfJetTracksAssociatorAtVertex'

pfTausPreSequence = cms.Sequence(
    pfJetTracksAssociatorAtVertex +
    pfTauPFJets08Region +
    pfTauPileUpVertices +
    pfTauTagInfoProducer
)

# Select taus from given collection that pass cloned discriminants
pfTaus = pfTauSelector.clone()
pfTaus.src = cms.InputTag("pfTausProducer")
pfTaus.discriminators = cms.VPSet(
        cms.PSet( discriminator=cms.InputTag("pfTausDiscriminationByDecayModeFinding"),selectionCut=cms.double(0.5) ),
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


