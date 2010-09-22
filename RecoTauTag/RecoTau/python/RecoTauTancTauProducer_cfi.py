import FWCore.ParameterSet.Config as cms

import RecoTauTag.RecoTau.RecoTauCleanerPlugins as cleaners
import RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingPionPtCut_cfi as leadPion_cfi 


'''

Configuration for TaNC taus

Decay mode is selected by the TaNC discriminator

'''

tancRecoTaus = cms.EDProducer(
    "RecoTauCleaner",
    tauSrc = cms.InputTag("combinatoricRecoTaus"),
    cleaners = cms.VPSet(
        cleaners.unitCharge,
        cleaners.leadPionFinding.clone(
            src = cms.InputTag("combinatoricRecoTausDiscriminationByLeadPionPtCut"),
        ),
        cleaners.tanc.clone(
            src = cms.InputTag("combinatoricRecoTausDiscriminationByTaNC"),
        ),
        # In case two taus both completely pass or fail tanc
        cleaners.chargeIsolation,
        cleaners.ecalIsolation,
    )
)

combinatoricRecoTausDiscriminationByLeadPionPtCut = \
        leadPion_cfi.pfRecoTauDiscriminationByLeadingPionPtCut.clone(
            PFTauProducer = cms.InputTag("combinatoricRecoTaus")
        )

combinatoricRecoTausDiscriminationByTaNC = cms.EDProducer(
    "RecoTauMVADiscriminator",
    PFTauProducer = cms.InputTag("combinatoricRecoTaus"),
    mvas = cms.VPSet(
        cms.PSet(
            nCharged = cms.uint32(1),
            nPiZeros = cms.uint32(0),
            mvaLabel = cms.string("OneProngNoPiZero"),
        ),
        cms.PSet(
            nCharged = cms.uint32(1),
            nPiZeros = cms.uint32(1),
            mvaLabel = cms.string("OneProngOnePiZero"),
        ),
        cms.PSet(
            nCharged = cms.uint32(1),
            nPiZeros = cms.uint32(2),
            mvaLabel = cms.string("OneProngTwoPiZero"),
        ),
        cms.PSet(
            nCharged = cms.uint32(3),
            nPiZeros = cms.uint32(0),
            mvaLabel = cms.string("ThreeProngNoPiZero"),
        ),
        cms.PSet(
            nCharged = cms.uint32(3),
            nPiZeros = cms.uint32(1),
            mvaLabel = cms.string("ThreeProngOnePiZero"),
        ),
    ),
    Prediscriminants = cms.PSet(
        BooleanOperator = cms.string("and"),
        leadPion = cms.PSet(
            cut = cms.double(0.5),
            Producer = cms.InputTag("combinatoricRecoTausDiscriminationByLeadPionPtCut")
        )
    ),
    prefailValue = cms.double(-2.0),
    dbLabel = cms.string(''),
)

tancRecoTausSequence = cms.Sequence(
    combinatoricRecoTausDiscriminationByLeadPionPtCut +
    combinatoricRecoTausDiscriminationByTaNC +
    tancRecoTaus 
)


