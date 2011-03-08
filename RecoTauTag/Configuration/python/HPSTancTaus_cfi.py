import FWCore.ParameterSet.Config as cms
import RecoTauTag.RecoTau.RecoTauCleanerPlugins as cleaners
from RecoTauTag.RecoTau.TauDiscriminatorTools import adaptTauDiscriminator, \
        producerIsTauTypeMapper
import copy

from RecoTauTag.RecoTau.hpstanc_transforms import transforms

# Apply the TaNC to the input tau collection
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingPionPtCut_cfi import \
        pfRecoTauDiscriminationByLeadingPionPtCut
# Common discrimination by lead pion
combinatoricRecoTausDiscriminationByLeadingPionPtCut = \
        pfRecoTauDiscriminationByLeadingPionPtCut.clone(
            PFTauProducer = cms.InputTag("combinatoricRecoTaus")
        )

# Steering file that loads the TaNC database, if necessary.  If unused it will
# be 'None'
from RecoTauTag.Configuration.RecoTauMVAConfiguration_cfi \
        import TauTagMVAComputerRecord

# Build the tanc discriminates
from RecoTauTag.RecoTau.RecoTauDiscriminantConfiguration import \
        discriminantConfiguration

combinatoricRecoTausDiscriminationByTanc = cms.EDProducer(
    "RecoTauMVADiscriminator",
    PFTauProducer = cms.InputTag("combinatoricRecoTaus"),
    Prediscriminants = noPrediscriminants,
    dbLabel = cms.string("hpstanc"),
    remapOutput = cms.bool(True),
    mvas = cms.VPSet(
        cms.PSet(
            nCharged = cms.uint32(1),
            nPiZeros = cms.uint32(0),
            mvaLabel = cms.string("1prong0pi0"),
        ),
        cms.PSet(
            nCharged = cms.uint32(1),
            nPiZeros = cms.uint32(1),
            mvaLabel = cms.string("1prong1pi0"),
        ),
        cms.PSet(
            nCharged = cms.uint32(1),
            nPiZeros = cms.uint32(2),
            mvaLabel = cms.string("1prong2pi0"),
        ),
        cms.PSet(
            nCharged = cms.uint32(3),
            nPiZeros = cms.uint32(0),
            mvaLabel = cms.string("3prong0pi0"),
        ),
    ),
    discriminantOptions = discriminantConfiguration,
)


#Produce the transformed TaNC output
combinatoricRecoTausTancTransform = cms.EDProducer(
    "RecoTauMVATransform",
    transforms = transforms, # blank for now
    PFTauProducer = cms.InputTag("combinatoricRecoTaus"),
    toTransform = cms.InputTag("combinatoricRecoTausDiscriminationByTanc"),
    Prediscriminants = noPrediscriminants
)

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByHPSSelection_cfi import \
        hpsSelectionDiscriminator

combinatoricRecoTausHPSSelector = hpsSelectionDiscriminator.clone(
    src = cms.InputTag("combinatoricRecoTaus"),
    minTauPt = cms.double(5.),
    # Turn off narrowness req.
    coneSizeFormula = cms.string('0.3'),
)

# Clean the taus according to the transformed output
hpsTancTaus = cms.EDProducer(
    "RecoTauCleaner",
    src = cms.InputTag("combinatoricRecoTaus"),
    cleaners = cms.VPSet(
        # Prefer taus that don't have charge == 3
        cleaners.unitCharge,
        # Prefer taus that are within DR<0.1 of the jet axis
        cleaners.matchingConeCut,
        # Prefer taus that pass the lead pion requirement
        cms.PSet(
            name = cms.string("lead pion"),
            plugin = cms.string("RecoTauDiscriminantCleanerPlugin"),
            src = cms.InputTag("combinatoricRecoTausDiscriminationByLeadingPionPtCut")
        ),
        # Prefer taus that pass the HPS selection
        cms.PSet(
            name = cms.string("HPS selection"),
            plugin = cms.string("RecoTauDiscriminantCleanerPlugin"),
            src = cms.InputTag("combinatoricRecoTausHPSSelector")
        ),
        cleaners.combinedIsolation
    )
)

# Rerun the leading pion cut on our clean taus
hpsTancTausDiscriminationByLeadingPionPtCut = \
        combinatoricRecoTausDiscriminationByLeadingPionPtCut.clone(
            PFTauProducer = cms.InputTag("hpsTancTaus"))

_leadPionPrediscriminant = cms.PSet(
    BooleanOperator = cms.string("and"),
    leadPion = cms.PSet(
        Producer = cms.InputTag(
            'hpsTancTausDiscriminationByLeadingPionPtCut'),
        cut = cms.double(0.5)
    )
)

# Build the HPS selection discriminator
hpsTancTausDiscriminationByDecayModeSelection = \
        combinatoricRecoTausHPSSelector.clone(
            PFTauProducer = cms.InputTag("hpsTancTaus"),
        )

from RecoTauTag.Configuration.HPSPFTaus_cfi import requireDecayMode,\
        hpsPFTauDiscriminationByVLooseIsolation,\
        hpsPFTauDiscriminationByLooseIsolation,\
        hpsPFTauDiscriminationByMediumIsolation,\
        hpsPFTauDiscriminationByTightIsolation

# Build the lead track and lepton discriminators
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackFinding_cfi import\
        pfRecoTauDiscriminationByLeadingTrackFinding
hpsTancTausDiscriminationByLeadingTrackFinding = copy.deepcopy(
    pfRecoTauDiscriminationByLeadingTrackFinding)
adaptTauDiscriminator(hpsTancTausDiscriminationByLeadingTrackFinding,
                      "hpsTancTaus", newTauTypeMapper=producerIsTauTypeMapper)

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackPtCut_cfi import \
        pfRecoTauDiscriminationByLeadingTrackPtCut
hpsTancTausDiscriminationByLeadingTrackPtCut = copy.deepcopy(
    pfRecoTauDiscriminationByLeadingTrackPtCut)
adaptTauDiscriminator(hpsTancTausDiscriminationByLeadingTrackPtCut,
                      "hpsTancTaus", newTauTypeMapper=producerIsTauTypeMapper)

from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectron_cfi import \
        pfRecoTauDiscriminationAgainstElectron
hpsTancTausDiscriminationAgainstElectron = copy.deepcopy(
    pfRecoTauDiscriminationAgainstElectron)
adaptTauDiscriminator(hpsTancTausDiscriminationAgainstElectron,
                      "hpsTancTaus", newTauTypeMapper=producerIsTauTypeMapper)

from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuon_cfi import \
        pfRecoTauDiscriminationAgainstMuon
hpsTancTausDiscriminationAgainstMuon = copy.deepcopy(
    pfRecoTauDiscriminationAgainstMuon)
adaptTauDiscriminator(hpsTancTausDiscriminationAgainstMuon,
                      "hpsTancTaus", newTauTypeMapper=producerIsTauTypeMapper)

from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstCaloMuon_cfi import \
        pfRecoTauDiscriminationAgainstCaloMuon
hpsTancTausDiscriminationAgainstCaloMuon = copy.deepcopy(
    pfRecoTauDiscriminationAgainstCaloMuon)
adaptTauDiscriminator(hpsTancTausDiscriminationAgainstCaloMuon,
                      "hpsTancTaus", newTauTypeMapper=producerIsTauTypeMapper)

# Update the decay mode prediscriminant
hpsTancRequireDecayMode = requireDecayMode.clone()
hpsTancRequireDecayMode.decayMode.Producer = cms.InputTag(
    "hpsTancTausDiscriminationByDecayModeSelection")

hpsTancTausDiscriminationByFlightPath = cms.EDProducer(
    "PFRecoTauDiscriminationByFlight",
    PFTauProducer = cms.InputTag("hpsTancTaus"),
    vertexSource = cms.InputTag("offlinePrimaryVertices"),
    beamspot = cms.InputTag("offlineBeamSpot"),
    refitPV = cms.bool(True),
    Prediscriminants = cms.PSet(
        BooleanOperator = cms.string("and"),
        leadTrack = cms.PSet(
            Producer = cms.InputTag(
                "hpsTancTausDiscriminationByDecayModeSelection"),
            cut = cms.double(0.5),
        )
    )
)

# Build the isolation discriminators
hpsTancTausDiscriminationByVLooseIsolation = \
        hpsPFTauDiscriminationByVLooseIsolation.clone(
            PFTauProducer = cms.InputTag("hpsTancTaus"),
            Prediscriminants = hpsTancRequireDecayMode
        )
hpsTancTausDiscriminationByLooseIsolation = \
        hpsPFTauDiscriminationByLooseIsolation.clone(
            PFTauProducer = cms.InputTag("hpsTancTaus"),
            Prediscriminants = hpsTancRequireDecayMode
        )
hpsTancTausDiscriminationByMediumIsolation = \
        hpsPFTauDiscriminationByMediumIsolation.clone(
            PFTauProducer = cms.InputTag("hpsTancTaus"),
            Prediscriminants = hpsTancRequireDecayMode
        )
hpsTancTausDiscriminationByTightIsolation = \
        hpsPFTauDiscriminationByTightIsolation.clone(
            PFTauProducer = cms.InputTag("hpsTancTaus"),
            Prediscriminants = hpsTancRequireDecayMode
        )

_tancPrediscriminants = _leadPionPrediscriminant.clone(
    hpsSelect = cms.PSet(
        Producer = cms.InputTag(
            'hpsTancTausDiscriminationByDecayModeSelection'),
        cut = cms.double(0.5)
    )
)

# Rerun the TaNC on our clean taus - in the future, rekey.
hpsTancTausDiscriminationByTancRaw = \
        combinatoricRecoTausDiscriminationByTanc.clone(
            PFTauProducer = cms.InputTag("hpsTancTaus"),
            Prediscriminants = _tancPrediscriminants,
        )

# Rerun the transformation
hpsTancTausDiscriminationByTanc = \
        combinatoricRecoTausTancTransform.clone(
            PFTauProducer = cms.InputTag("hpsTancTaus"),
            toTransform = cms.InputTag("hpsTancTausDiscriminationByTancRaw"),
            transforms = transforms,
            Prediscriminants = _tancPrediscriminants,
        )

hpsTancTausDiscriminationByTancLoose = cms.EDProducer(
    # There is no PT selection applied; we use this only to use the
    # prediscriminants to make binary cuts.
    "PFRecoTauDiscriminationByLeadingObjectPtCut",
    PFTauProducer = cms.InputTag("hpsTancTaus"),
    UseOnlyChargedHadrons = cms.bool(True),
    MinPtLeadingObject = cms.double(0.0),
    Prediscriminants = cms.PSet(
        BooleanOperator = cms.string("and"),
        tancCut = cms.PSet(
            Producer = cms.InputTag("hpsTancTausDiscriminationByTanc"),
            cut = cms.double(0.95),
        )
    )
)

# Make a very loose cut
hpsTancTausDiscriminationByTancVLoose = \
        hpsTancTausDiscriminationByTancLoose.clone()
hpsTancTausDiscriminationByTancVLoose.Prediscriminants.tancCut.cut = 0.90

hpsTancTausDiscriminationByTancMedium = \
        hpsTancTausDiscriminationByTancLoose.clone()
hpsTancTausDiscriminationByTancMedium.Prediscriminants.tancCut.cut = 0.97

hpsTancTausDiscriminationByTancTight = \
        hpsTancTausDiscriminationByTancLoose.clone()
hpsTancTausDiscriminationByTancTight.Prediscriminants.tancCut.cut = 0.985

hpsTancTauInitialSequence = cms.Sequence(
    combinatoricRecoTausDiscriminationByLeadingPionPtCut
    + combinatoricRecoTausHPSSelector
    # select & clean each decay mode
    + hpsTancTaus
    + hpsTancTausDiscriminationByLeadingTrackFinding
    + hpsTancTausDiscriminationByLeadingPionPtCut
    + hpsTancTausDiscriminationByLeadingTrackPtCut
    + hpsTancTausDiscriminationByDecayModeSelection
    + hpsTancTausDiscriminationByFlightPath
)

hpsTancTauDiscriminantSequence = cms.Sequence(
    hpsTancTausDiscriminationAgainstElectron
    + hpsTancTausDiscriminationAgainstMuon
    #+ hpsTancTausDiscriminationAgainstCaloMuon
    + hpsTancTausDiscriminationByTancRaw
    + hpsTancTausDiscriminationByTanc
    + hpsTancTausDiscriminationByTancVLoose
    + hpsTancTausDiscriminationByTancLoose
    + hpsTancTausDiscriminationByTancMedium
    + hpsTancTausDiscriminationByTancTight
    + hpsTancTausDiscriminationByVLooseIsolation
    + hpsTancTausDiscriminationByLooseIsolation
    + hpsTancTausDiscriminationByMediumIsolation
    + hpsTancTausDiscriminationByTightIsolation
)
