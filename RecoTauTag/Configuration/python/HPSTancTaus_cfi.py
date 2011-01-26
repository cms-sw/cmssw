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

# Eventually this will come from the global tag
#from RecoTauTag.TauTagTools.TancConditions_cff import TauTagMVAComputerRecord
#TauTagMVAComputerRecord.connect = cms.string(
    #'sqlite_fip:RecoTauTag/RecoTau/data/hpstanc.db'
#)
#TauTagMVAComputerRecord.toGet[0].tag = cms.string('Tanc')
## Don't conflict with TaNC global tag
#TauTagMVAComputerRecord.appendToDataLabel = cms.string('hpstanc')

# Build the tanc discriminates
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
    )
)

combinatoricTaus1prong0pi0 = cms.EDFilter(
    "PFTauViewRefSelector",
    src = cms.InputTag("combinatoricRecoTaus"),
    cut = cms.string("decayMode = 0"),
)

combinatoricTaus1prong1pi0 = cms.EDFilter(
    "PFTauViewRefSelector",
    src = cms.InputTag("combinatoricRecoTaus"),
    cut = cms.string("decayMode = 1"),
)

combinatoricTaus1prong2pi0 = cms.EDFilter(
    "PFTauViewRefSelector",
    src = cms.InputTag("combinatoricRecoTaus"),
    cut = cms.string("decayMode = 2"),
)

combinatoricTaus3prong0pi0 = cms.EDFilter(
    "PFTauViewRefSelector",
    src = cms.InputTag("combinatoricRecoTaus"),
    cut = cms.string("decayMode = 10"),
)

# Clean each collection separately using TaNC
selected1prong0pi0TancTaus = cms.EDProducer(
    "RecoTauRefCleaner",
    src = cms.InputTag("combinatoricTaus1prong0pi0"),
    cleaners = cms.VPSet(
        # Prefer taus that don't have charge == 3
        cleaners.unitCharge,
        # Prefer taus that pass the lead pion requirement
        cms.PSet(
            name = cms.string("lead pion"),
            plugin = cms.string("RecoTauDiscriminantCleanerPlugin"),
            src = cms.InputTag(
                "combinatoricRecoTausDiscriminationByLeadingPionPtCut")
        ),
        # Finally rank taus according to their transformed TaNC output
        cms.PSet(
            name = cms.string("TaNC transform"),
            plugin = cms.string("RecoTauDiscriminantCleanerPlugin"),
            src = cms.InputTag("combinatoricRecoTausDiscriminationByTanc")
        ),
    )
)
selected1prong1pi0TancTaus = selected1prong0pi0TancTaus.clone(
    src = cms.InputTag("combinatoricTaus1prong1pi0")
)
selected1prong2pi0TancTaus = selected1prong0pi0TancTaus.clone(
    src = cms.InputTag("combinatoricTaus1prong2pi0")
)
selected3prong0pi0TancTaus = selected1prong0pi0TancTaus.clone(
    src = cms.InputTag("combinatoricTaus3prong0pi0")
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
    src = cms.InputTag("combinatoricRecoTaus"))

# Merge are the decay modes back together.  Each decay mode has had it's best
# candidate selected.
hpsTancTausDecayModeClean = cms.EDProducer(
    "PFTauViewRefMerger",
    src = cms.VInputTag(
        "selected1prong0pi0TancTaus",
        "selected1prong1pi0TancTaus",
        "selected1prong2pi0TancTaus",
        "selected3prong0pi0TancTaus",
    )
)

# Clean the taus according to the transformed output
hpsTancTaus = cms.EDProducer(
    "RecoTauCleaner",
    src = cms.InputTag("hpsTancTausDecayModeClean"),
    cleaners = cms.VPSet(
        # Prefer taus that don't have charge == 3
        cleaners.unitCharge,
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
        # Finally rank taus according to their transformed TaNC output
        #cms.PSet(
            #name = cms.string("TaNC transform"),
            #plugin = cms.string("RecoTauDiscriminantCleanerPlugin"),
            #src = cms.InputTag("combinatoricRecoTausTancTransform")
        #),
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
hpsTancTausDiscriminationByDecayModeSelection = hpsSelectionDiscriminator.clone(
    PFTauProducer = cms.InputTag("hpsTancTaus"),
)

from RecoTauTag.Configuration.HPSPFTaus_cfi import requireDecayMode,\
        hpsPFTauDiscriminationByLooseIsolation,\
        hpsPFTauDiscriminationByMediumIsolation,\
        hpsPFTauDiscriminationByTightIsolation
        #hpsPFTauDiscriminationByVLooseIsolation,\

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

# Build the isolation discriminators
#hpsTancTausDiscriminationByVLooseIsolation = \
        #hpsPFTauDiscriminationByVLooseIsolation.clone(
            #PFTauProducer = cms.InputTag("hpsTancTaus"),
            #Prediscriminants = hpsTancRequireDecayMode
        #)
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
            transforms = transforms,
            Prediscriminants = _leadPionPrediscriminant
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

hpsTancTauSequence = cms.Sequence(
    combinatoricRecoTausDiscriminationByTanc
    + combinatoricRecoTausDiscriminationByLeadingPionPtCut
    + combinatoricRecoTausTancTransform
    + combinatoricRecoTausHPSSelector
    # select & clean each decay mode
    + combinatoricTaus1prong0pi0
    + combinatoricTaus1prong1pi0
    + combinatoricTaus1prong2pi0
    + combinatoricTaus3prong0pi0
    + selected1prong0pi0TancTaus
    + selected1prong1pi0TancTaus
    + selected1prong2pi0TancTaus
    + selected3prong0pi0TancTaus
    + hpsTancTausDecayModeClean
    + hpsTancTaus
    + hpsTancTausDiscriminationByLeadingTrackFinding
    + hpsTancTausDiscriminationByLeadingPionPtCut
    + hpsTancTausDiscriminationByLeadingTrackPtCut
    + hpsTancTausDiscriminationAgainstElectron
    + hpsTancTausDiscriminationAgainstMuon
    #+ hpsTancTausDiscriminationAgainstCaloMuon
    + hpsTancTausDiscriminationByTancRaw
    + hpsTancTausDiscriminationByTanc
    + hpsTancTausDiscriminationByTancVLoose
    + hpsTancTausDiscriminationByTancLoose
    + hpsTancTausDiscriminationByTancMedium
    + hpsTancTausDiscriminationByTancTight
    + hpsTancTausDiscriminationByDecayModeSelection
    #+ hpsTancTausDiscriminationByVLooseIsolation
    + hpsTancTausDiscriminationByLooseIsolation
    + hpsTancTausDiscriminationByMediumIsolation
    + hpsTancTausDiscriminationByTightIsolation
)
