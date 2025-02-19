import FWCore.ParameterSet.Config as cms
import RecoTauTag.RecoTau.RecoTauCleanerPlugins as cleaners
from RecoTauTag.RecoTau.TauDiscriminatorTools import adaptTauDiscriminator, producerIsTauTypeMapper
from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts
import copy

from RecoTauTag.RecoTau.hpstanc_transforms import transforms, cuts

# Apply the TaNC to the input tau collection
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingPionPtCut_cfi import pfRecoTauDiscriminationByLeadingPionPtCut
# Common discrimination by lead pion
combinatoricRecoTausDiscriminationByLeadingPionPtCut = pfRecoTauDiscriminationByLeadingPionPtCut.clone(
            PFTauProducer = cms.InputTag("combinatoricRecoTaus")
        )

# Steering file that loads the TaNC database, if necessary.  If unused it will
# be 'None'
from RecoTauTag.Configuration.RecoTauMVAConfiguration_cfi import TauTagMVAComputerRecord, esPreferLocalTancDB

# Build the tanc discriminates
from RecoTauTag.RecoTau.RecoTauDiscriminantConfiguration import discriminantConfiguration

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

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByHPSSelection_cfi import hpsSelectionDiscriminator

combinatoricRecoTausHPSSelector = hpsSelectionDiscriminator.clone(
    src = cms.InputTag("combinatoricRecoTaus"),
    minTauPt = cms.double(5.),
    # Turn off narrowness req.
    coneSizeFormula = cms.string('0.3'),
)

# Clean the taus according to the transformed output
hpsTancTausSansRefs = cms.EDProducer(
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

hpsTancTaus = cms.EDProducer(
    "RecoTauPiZeroUnembedder",
    src = cms.InputTag("hpsTancTausSansRefs")
)

# Rerun the leading pion cut on our clean taus
hpsTancTausDiscriminationByLeadingPionPtCut = combinatoricRecoTausDiscriminationByLeadingPionPtCut.clone(
    PFTauProducer = cms.InputTag("hpsTancTaus")
    )

_leadPionPrediscriminant = cms.PSet(
    BooleanOperator = cms.string("and"),
    leadPion = cms.PSet(
        Producer = cms.InputTag(
            'hpsTancTausDiscriminationByLeadingPionPtCut'),
        cut = cms.double(0.5)
    )
)

# Build the HPS selection discriminator
hpsTancTausDiscriminationByDecayModeSelection = combinatoricRecoTausHPSSelector.clone(
    PFTauProducer = cms.InputTag("hpsTancTaus")
    )

from RecoTauTag.Configuration.HPSPFTaus_cff import requireDecayMode,\
     hpsPFTauDiscriminationByVLooseIsolation,\
     hpsPFTauDiscriminationByLooseIsolation,\
     hpsPFTauDiscriminationByMediumIsolation,\
     hpsPFTauDiscriminationByTightIsolation

# Build the lead track and lepton discriminators
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackFinding_cfi import pfRecoTauDiscriminationByLeadingTrackFinding
hpsTancTausDiscriminationByLeadingTrackFinding = pfRecoTauDiscriminationByLeadingTrackFinding.clone(
    PFTauProducer = cms.InputTag("hpsTancTaus")
    )

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackPtCut_cfi import pfRecoTauDiscriminationByLeadingTrackPtCut
hpsTancTausDiscriminationByLeadingTrackPtCut = pfRecoTauDiscriminationByLeadingTrackPtCut.clone(
    PFTauProducer = cms.InputTag("hpsTancTaus")
    )


# Build lepton discriminants
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectron_cfi import pfRecoTauDiscriminationAgainstElectron

hpsTancTausDiscriminationByLooseElectronRejection = pfRecoTauDiscriminationAgainstElectron.clone(
    PFTauProducer = cms.InputTag("hpsTancTaus"),
    Prediscriminants = noPrediscriminants,
    PFElectronMVA_maxValue = cms.double(0.6)
    )

hpsTancTausDiscriminationByMediumElectronRejection = pfRecoTauDiscriminationAgainstElectron.clone(
    PFTauProducer = cms.InputTag("hpsTancTaus"),
    Prediscriminants = noPrediscriminants,
    ApplyCut_EcalCrackCut = cms.bool(True)
    )

hpsTancTausDiscriminationByTightElectronRejection = pfRecoTauDiscriminationAgainstElectron.clone(
    PFTauProducer = cms.InputTag("hpsTancTaus"),
    Prediscriminants = noPrediscriminants,
    ApplyCut_EcalCrackCut = cms.bool(True),
    ApplyCut_BremCombined = cms.bool(True)
    )

from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuon_cfi import pfRecoTauDiscriminationAgainstMuon

hpsTancTausDiscriminationByLooseMuonRejection = pfRecoTauDiscriminationAgainstMuon.clone(
    PFTauProducer = cms.InputTag("hpsTancTaus"),
    Prediscriminants = noPrediscriminants
   )

hpsTancTausDiscriminationByTightMuonRejection = pfRecoTauDiscriminationAgainstMuon.clone(
    PFTauProducer = cms.InputTag("hpsTancTaus"),
    Prediscriminants = noPrediscriminants,
    discriminatorOption = cms.string('noAllArbitrated')
    )

from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstCaloMuon_cfi import pfRecoTauDiscriminationAgainstCaloMuon
hpsTancTausDiscriminationAgainstCaloMuon = pfRecoTauDiscriminationAgainstCaloMuon.clone(
    PFTauProducer = cms.InputTag("hpsTancTaus")
    )
hpsTancTausDiscriminationAgainstCaloMuon.Prediscriminants.leadTrack.Producer = cms.InputTag("hpsTancTausDiscriminationByLeadingTrackFinding")

# Update the decay mode prediscriminant
hpsTancRequireDecayMode = requireDecayMode.clone()
hpsTancRequireDecayMode.decayMode.Producer = cms.InputTag("hpsTancTausDiscriminationByDecayModeSelection")

hpsTancTausDiscriminationByFlightPath = cms.EDProducer(
    "PFRecoTauDiscriminationByFlight",
    PFTauProducer = cms.InputTag("hpsTancTaus"),
    vertexSource = PFTauQualityCuts.primaryVertexSrc,
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
hpsTancTausDiscriminationByVLooseIsolation.Prediscriminants = hpsTancRequireDecayMode

hpsTancTausDiscriminationByLooseIsolation = \
        hpsPFTauDiscriminationByLooseIsolation.clone(
            PFTauProducer = cms.InputTag("hpsTancTaus"),
            Prediscriminants = hpsTancRequireDecayMode
        )
hpsTancTausDiscriminationByLooseIsolation.Prediscriminants = hpsTancRequireDecayMode

hpsTancTausDiscriminationByMediumIsolation = \
        hpsPFTauDiscriminationByMediumIsolation.clone(
            PFTauProducer = cms.InputTag("hpsTancTaus"),
            Prediscriminants = hpsTancRequireDecayMode
        )
hpsTancTausDiscriminationByMediumIsolation.Prediscriminants = hpsTancRequireDecayMode

hpsTancTausDiscriminationByTightIsolation = \
        hpsPFTauDiscriminationByTightIsolation.clone(
            PFTauProducer = cms.InputTag("hpsTancTaus"),
            Prediscriminants = hpsTancRequireDecayMode
        )
hpsTancTausDiscriminationByTightIsolation.Prediscriminants = hpsTancRequireDecayMode

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
            cut = cuts.looseCut
        )
    )
)

# Make a very loose cut
hpsTancTausDiscriminationByTancVLoose = \
        hpsTancTausDiscriminationByTancLoose.clone()
hpsTancTausDiscriminationByTancVLoose.Prediscriminants.tancCut.cut = cuts.vlooseCut

hpsTancTausDiscriminationByTancMedium = \
        hpsTancTausDiscriminationByTancLoose.clone()
hpsTancTausDiscriminationByTancMedium.Prediscriminants.tancCut.cut = cuts.mediumCut

hpsTancTausDiscriminationByTancTight = \
        hpsTancTausDiscriminationByTancLoose.clone()
hpsTancTausDiscriminationByTancTight.Prediscriminants.tancCut.cut = cuts.tightCut

hpsTancTauInitialSequence = cms.Sequence(
    combinatoricRecoTausDiscriminationByLeadingPionPtCut
    + combinatoricRecoTausHPSSelector
    # select & clean each decay mode
    + hpsTancTausSansRefs
    + hpsTancTaus
    + hpsTancTausDiscriminationByLeadingTrackFinding
    + hpsTancTausDiscriminationByLeadingPionPtCut
    + hpsTancTausDiscriminationByLeadingTrackPtCut
    + hpsTancTausDiscriminationByDecayModeSelection
    #+ hpsTancTausDiscriminationByFlightPath
)

hpsTancTauDiscriminantSequence = cms.Sequence(
    #+ hpsTancTausDiscriminationAgainstCaloMuon
    hpsTancTausDiscriminationByTancRaw
    + hpsTancTausDiscriminationByTanc
    + hpsTancTausDiscriminationByTancVLoose
    + hpsTancTausDiscriminationByTancLoose
    + hpsTancTausDiscriminationByTancMedium
    + hpsTancTausDiscriminationByTancTight
    + hpsTancTausDiscriminationByVLooseIsolation
    + hpsTancTausDiscriminationByLooseIsolation
    + hpsTancTausDiscriminationByMediumIsolation
    + hpsTancTausDiscriminationByTightIsolation
    + hpsTancTausDiscriminationByLooseElectronRejection
    + hpsTancTausDiscriminationByMediumElectronRejection
    + hpsTancTausDiscriminationByTightElectronRejection
    + hpsTancTausDiscriminationByLooseMuonRejection
    + hpsTancTausDiscriminationByTightMuonRejection
)
