import FWCore.ParameterSet.Config as cms
import RecoTauTag.RecoTau.RecoTauCleanerPlugins as cleaners

TANC_TRANSFORM = cms.VPSet()

# Apply the TaNC to the input tau collection
from RecoTauTag.RecoTau.TauDiscriminatorTools import noPrediscriminants

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingPionPtCut_cfi import \
        pfRecoTauDiscriminationByLeadingPionPtCut
# Common discrimination by lead pion
combinatoricRecoTausDiscriminationByLeadingPionPtCut = \
        pfRecoTauDiscriminationByLeadingPionPtCut.clone(
            PFTauProducer = cms.InputTag("combinatoricRecoTaus")
        )

# Build the tanc discriminates
combinatoricRecoTausDiscriminationByTanc = cms.EDProducer(
    "RecoTauMVADiscriminator",
    PFTauProducer = cms.InputTag("combinatoricRecoTaus"),
    Prediscriminants = noPrediscriminants,
    dbLabel = cms.string(""),
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
    )
)

# Produce the transformed TaNC output
combinatoricRecoTausTancTransform = cms.EDProducer(
    "RecoTauMVATransform",
    transforms = TANC_TRANSFORM, # blank for now
    PFTauProducer = cms.InputTag("combinatoricRecoTaus"),
    toTransform = cms.InputTag("combinatoricRecoTausDiscriminationByTanc"),
    Prediscriminants = noPrediscriminants
)

#from RecoTauTag.RecoTau.PFRecoTauDiscriminationByHPSSelection_cfi import \
#        hpsSelectionDiscriminator
#combinatoricRecoTausHPSSelector = hpsSelectionDiscriminator.clone(
#    src = cms.InputTag("hpsPFTauDiscriminationAgainstMuon


# Clean the taus according to the transformed output
hpsTancTaus = cms.EDProducer(
    "RecoTauCleaner",
    src = cms.InputTag("combinatoricRecoTaus"),
    cleaners = cms.VPSet(
        # Prefer taus that don't have charge == 3
        cleaners.unitCharge,
        # Prefer taus that pass the lead pion requirement
        cms.PSet(
            name = cms.string("lead pion"),
            plugin = cms.string("RecoTauDiscriminantCleanerPlugin"),
            src = cms.InputTag("combinatoricRecoTausDiscriminationByLeadingPionPtCut")
        ),
        # Finally rank taus according to their transformed TaNC output
        cms.PSet(
            name = cms.string("TaNC transform"),
            plugin = cms.string("RecoTauDiscriminantCleanerPlugin"),
            src = cms.InputTag("combinatoricRecoTausTancTransform")
            #src = cms.InputTag("combinatoricRecoTausDiscriminationByTanc")
        ),
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

# Rerun the TaNC on our clean taus - in the future, rekey.
hpsTancTausDiscriminationByTancRaw = \
        combinatoricRecoTausDiscriminationByTanc.clone(
            PFTauProducer = cms.InputTag("hpsTancTaus"),
            Prediscriminants = _leadPionPrediscriminant
        )

# Rerun the transformation
hpsTancTausDiscriminationByTanc = \
        combinatoricRecoTausTancTransform.clone(
            PFTauProducer = cms.InputTag("hpsTancTaus"),
            toTransform = cms.InputTag("hpsTancTausDiscriminationByTancRaw"),
            transforms = TANC_TRANSFORM,
            Prediscriminants = _leadPionPrediscriminant
        )

hpsTancTauSequence = cms.Sequence(
    combinatoricRecoTausDiscriminationByTanc
    + combinatoricRecoTausDiscriminationByLeadingPionPtCut
    + combinatoricRecoTausTancTransform
    + hpsTancTaus
    + hpsTancTausDiscriminationByLeadingPionPtCut
    + hpsTancTausDiscriminationByTancRaw
    + hpsTancTausDiscriminationByTanc
)
