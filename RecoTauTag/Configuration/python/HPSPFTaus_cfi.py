import FWCore.ParameterSet.Config as cms
import copy

'''

Sequences for HPS taus

'''

# Define the discriminators for this tau
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi                      import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackFinding_cfi            import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectron_cfi                  import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuon_cfi                      import *

# Load helper functions to change the source of the discriminants
from RecoTauTag.RecoTau.TauDiscriminatorTools import *

# Select those taus that pass the HPS selections
#  - pt > 15, mass cuts, tauCone cut
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByHPSSelection_cfi import \
        hpsSelectionDiscriminator
hpsPFTauDiscriminationByDecayModeFinding = copy.deepcopy(
    hpsSelectionDiscriminator)
hpsPFTauDiscriminationByDecayModeFinding.PFTauProducer \
        = cms.InputTag('hpsPFTauProducer')

#setTauSource(hpsPFTauDiscriminationByDecayModeFinding, 'hpsPFTauProducer')
# Define decay mode prediscriminant
requireDecayMode = cms.PSet(
    BooleanOperator = cms.string("and"),
    decayMode = cms.PSet(
        Producer = cms.InputTag('hpsPFTauDiscriminationByDecayModeFinding'),
        cut = cms.double(0.5)
    )
)

#copying the Discriminator by Isolation
hpsPFTauDiscriminationByLooseIsolation = copy.deepcopy(pfRecoTauDiscriminationByIsolation)
setTauSource(hpsPFTauDiscriminationByLooseIsolation, 'hpsPFTauProducer')
hpsPFTauDiscriminationByLooseIsolation.Prediscriminants = requireDecayMode

#Define a discriminator By Medium Isolation!
#You need to loosen qualityCuts for this
mediumPFTauQualityCuts = cms.PSet(
      signalQualityCuts = cms.PSet(
         minTrackPt                   = cms.double(0.8),  # filter PFChargedHadrons below given pt
         maxTrackChi2                 = cms.double(100.), # require track Chi2
         maxTransverseImpactParameter = cms.double(0.03), # w.r.t. PV
         maxDeltaZ                    = cms.double(0.2),  # w.r.t. PV
         minTrackPixelHits            = cms.uint32(0),    # pixel-only hits (note that these cuts are turned off,
                                                          # the tracking cuts might be higher)
         minTrackHits                 = cms.uint32(3),    # total track hits
         minGammaEt                   = cms.double(0.5),  # filter PFgammas below given Pt
         useTracksInsteadOfPFHadrons  = cms.bool(False),  # if true, use generalTracks, instead of PFChargedHadrons
      ),
      isolationQualityCuts = cms.PSet(
         minTrackPt                   = cms.double(0.8),
         maxTrackChi2                 = cms.double(100.),
         maxTransverseImpactParameter = cms.double(0.03),
         maxDeltaZ                    = cms.double(0.2),
         minTrackPixelHits            = cms.uint32(0),
         minTrackHits                 = cms.uint32(3),
         minGammaEt                   = cms.double(0.8),
         useTracksInsteadOfPFHadrons  = cms.bool(False),
      )
)

hpsPFTauDiscriminationByMediumIsolation = copy.deepcopy(pfRecoTauDiscriminationByIsolation)
setTauSource(hpsPFTauDiscriminationByMediumIsolation, 'hpsPFTauProducer')
hpsPFTauDiscriminationByMediumIsolation.Prediscriminants = requireDecayMode
hpsPFTauDiscriminationByMediumIsolation.qualityCuts = mediumPFTauQualityCuts

#Define a discriminator By Tight Isolation!
#You need to loosen qualityCuts for this
loosePFTauQualityCuts = cms.PSet(
      signalQualityCuts = cms.PSet(
         minTrackPt                   = cms.double(0.5),  # filter PFChargedHadrons below given pt
         maxTrackChi2                 = cms.double(100.), # require track Chi2
         maxTransverseImpactParameter = cms.double(0.03), # w.r.t. PV
         maxDeltaZ                    = cms.double(0.2),  # w.r.t. PV
         minTrackPixelHits            = cms.uint32(0),    # pixel-only hits (note that these cuts are turned off,
                                                          # the tracking cuts might be higher)
         minTrackHits                 = cms.uint32(3),    # total track hits
         minGammaEt                   = cms.double(0.5),  # filter PFgammas below given Pt
         useTracksInsteadOfPFHadrons  = cms.bool(False),  # if true, use generalTracks, instead of PFChargedHadrons
      ),
      isolationQualityCuts = cms.PSet(
         minTrackPt                   = cms.double(0.5),
         maxTrackChi2                 = cms.double(100.),
         maxTransverseImpactParameter = cms.double(0.03),
         maxDeltaZ                    = cms.double(0.2),
         minTrackPixelHits            = cms.uint32(0),
         minTrackHits                 = cms.uint32(3),
         minGammaEt                   = cms.double(0.5),
         useTracksInsteadOfPFHadrons  = cms.bool(False),
      )
)

hpsPFTauDiscriminationByTightIsolation = copy.deepcopy(pfRecoTauDiscriminationByIsolation)
setTauSource(hpsPFTauDiscriminationByTightIsolation, 'hpsPFTauProducer')
hpsPFTauDiscriminationByTightIsolation.Prediscriminants = requireDecayMode
hpsPFTauDiscriminationByTightIsolation.qualityCuts = loosePFTauQualityCuts# set the standard quality cuts

#copying discriminator against electrons and muons
hpsPFTauDiscriminationAgainstElectron = copy.deepcopy(pfRecoTauDiscriminationAgainstElectron)
setTauSource(hpsPFTauDiscriminationAgainstElectron, 'hpsPFTauProducer')
hpsPFTauDiscriminationAgainstElectron.Prediscriminants = noPrediscriminants

hpsPFTauDiscriminationAgainstMuon = copy.deepcopy(pfRecoTauDiscriminationAgainstMuon)
setTauSource(hpsPFTauDiscriminationAgainstMuon, 'hpsPFTauProducer')
hpsPFTauDiscriminationAgainstMuon.Prediscriminants = noPrediscriminants

# Define the HPS selection discriminator used in cleaning
hpsSelectionDiscriminator.PFTauProducer = cms.InputTag("combinatoricRecoTaus")

# Define discriminants to use for HPS cleaning
hpsTightIsolationCleaner = hpsPFTauDiscriminationByTightIsolation.clone(
    Prediscriminants = noPrediscriminants,
    PFTauProducer = cms.InputTag("combinatoricRecoTaus"),
)
hpsMediumIsolationCleaner = hpsPFTauDiscriminationByMediumIsolation.clone(
    Prediscriminants = noPrediscriminants,
    PFTauProducer = cms.InputTag("combinatoricRecoTaus"),
)
hpsLooseIsolationCleaner = hpsPFTauDiscriminationByLooseIsolation.clone(
    Prediscriminants = noPrediscriminants,
    PFTauProducer = cms.InputTag("combinatoricRecoTaus"),
)

import RecoTauTag.RecoTau.RecoTauCleanerPlugins as cleaners

hpsPFTauProducer = cms.EDProducer(
    "RecoTauCleaner",
    src = cms.InputTag("combinatoricRecoTaus"),
    cleaners = cms.VPSet(
        # Prefer taus that dont' have charge == 3
        cleaners.unitCharge,
        # Prefer taus that pass HPS selections
        cms.PSet(
            name = cms.string("HPS_Select"),
            plugin = cms.string("RecoTauDiscriminantCleanerPlugin"),
            src = cms.InputTag("hpsSelectionDiscriminator"),
        ),
        # Then prefer those that pass isolation.  Also prefer those that pass
        # the tighter isolations
        cms.PSet(
            name = cms.string("TightIso"),
            plugin = cms.string("RecoTauDiscriminantCleanerPlugin"),
            src = cms.InputTag("hpsTightIsolationCleaner"),
        ),
        cms.PSet(
            name = cms.string("MediumIso"),
            plugin = cms.string("RecoTauDiscriminantCleanerPlugin"),
            src = cms.InputTag("hpsMediumIsolationCleaner"),
        ),
        cms.PSet(
            name = cms.string("LooseIso"),
            plugin = cms.string("RecoTauDiscriminantCleanerPlugin"),
            src = cms.InputTag("hpsLooseIsolationCleaner"),
        ),
        # Finally, if all this passes, take the one with less stuff in the
        # isolation cone.
        cleaners.combinedIsolation
    )
)

produceHPSPFTaus = cms.Sequence(
    hpsSelectionDiscriminator
    *hpsTightIsolationCleaner
    *hpsMediumIsolationCleaner
    *hpsLooseIsolationCleaner
    *hpsPFTauProducer
)

produceAndDiscriminateHPSPFTaus = cms.Sequence(
    produceHPSPFTaus*
    hpsPFTauDiscriminationByDecayModeFinding*
    hpsPFTauDiscriminationByLooseIsolation*
    hpsPFTauDiscriminationByMediumIsolation*
    hpsPFTauDiscriminationByTightIsolation*
    hpsPFTauDiscriminationAgainstElectron*
    hpsPFTauDiscriminationAgainstMuon
)
