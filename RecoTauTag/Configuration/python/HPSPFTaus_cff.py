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

#Building the prototype for  the Discriminator by Isolation
hpsPFTauDiscriminationByLooseIsolation = copy.deepcopy(pfRecoTauDiscriminationByIsolation)
setTauSource(hpsPFTauDiscriminationByLooseIsolation, 'hpsPFTauProducer')
hpsPFTauDiscriminationByLooseIsolation.Prediscriminants = requireDecayMode.clone()

# First apply only charged isolation
hpsPFTauDiscriminationByLooseChargedIsolation = hpsPFTauDiscriminationByLooseIsolation.clone()
hpsPFTauDiscriminationByLooseChargedIsolation.ApplyDiscriminationByECALIsolation = False
hpsPFTauDiscriminationByLooseChargedIsolation.applyDeltaBetaCorrection = False

hpsPFTauDiscriminationByLooseIsolation.ApplyDiscriminationByTrackerIsolation = False
hpsPFTauDiscriminationByLooseIsolation.ApplyDiscriminationByECALIsolation = True
hpsPFTauDiscriminationByLooseIsolation.applyDeltaBetaCorrection = True
# Instead of occupancy, we use sum pt so the DeltaBeta correction is continuous
hpsPFTauDiscriminationByLooseIsolation.applyOccupancyCut = False
hpsPFTauDiscriminationByLooseIsolation.applySumPtCut = True
hpsPFTauDiscriminationByLooseIsolation.maximumSumPtCut = \
        hpsPFTauDiscriminationByLooseIsolation.qualityCuts.\
        isolationQualityCuts.minGammaEt
# Apply the charged prediscriminant before computed the ECAL isolation
hpsPFTauDiscriminationByLooseIsolation.Prediscriminants.preIso = cms.PSet(
    Producer = cms.InputTag("hpsPFTauDiscriminationByLooseChargedIsolation"),
    cut = cms.double(0.5)
)

# Make an even looser discriminator
hpsPFTauDiscriminationByVLooseChargedIsolation = hpsPFTauDiscriminationByLooseChargedIsolation.clone()
hpsPFTauDiscriminationByVLooseChargedIsolation.qualityCuts.isolationQualityCuts.minTrackPt = 1.5
hpsPFTauDiscriminationByVLooseChargedIsolation.qualityCuts.isolationQualityCuts.minGammaEt = 2.0
hpsPFTauDiscriminationByVLooseChargedIsolation.customOuterCone = cms.double(0.3)
hpsPFTauDiscriminationByVLooseChargedIsolation.isoConeSizeForDeltaBeta = cms.double(0.3)

hpsPFTauDiscriminationByVLooseIsolation = hpsPFTauDiscriminationByLooseIsolation.clone()
hpsPFTauDiscriminationByVLooseIsolation.qualityCuts.isolationQualityCuts.minTrackPt = 1.5
hpsPFTauDiscriminationByVLooseIsolation.qualityCuts.isolationQualityCuts.minGammaEt = 2.0
hpsPFTauDiscriminationByVLooseIsolation.maximumSumPtCut = \
        hpsPFTauDiscriminationByVLooseIsolation.qualityCuts.\
        isolationQualityCuts.minGammaEt

hpsPFTauDiscriminationByVLooseIsolation.customOuterCone = cms.double(0.3)
hpsPFTauDiscriminationByVLooseIsolation.isoConeSizeForDeltaBeta = cms.double(0.3)
hpsPFTauDiscriminationByVLooseIsolation.Prediscriminants.preIso.Producer = \
        cms.InputTag("hpsPFTauDiscriminationByVLooseChargedIsolation")

hpsPFTauDiscriminationByMediumChargedIsolation = hpsPFTauDiscriminationByLooseChargedIsolation.clone()
hpsPFTauDiscriminationByMediumChargedIsolation.qualityCuts.isolationQualityCuts.minTrackPt = 0.8
hpsPFTauDiscriminationByMediumChargedIsolation.qualityCuts.isolationQualityCuts.minGammaEt = 0.8
hpsPFTauDiscriminationByMediumChargedIsolation.Prediscriminants.preIso = cms.PSet(
    Producer = cms.InputTag("hpsPFTauDiscriminationByLooseChargedIsolation"),
    cut = cms.double(0.5)
)

hpsPFTauDiscriminationByMediumIsolation = hpsPFTauDiscriminationByLooseIsolation.clone()
hpsPFTauDiscriminationByMediumIsolation.qualityCuts.isolationQualityCuts.minTrackPt = 0.8
hpsPFTauDiscriminationByMediumIsolation.qualityCuts.isolationQualityCuts.minGammaEt = 0.8
hpsPFTauDiscriminationByMediumIsolation.maximumSumPtCut = \
        hpsPFTauDiscriminationByMediumIsolation.qualityCuts.\
        isolationQualityCuts.minGammaEt
hpsPFTauDiscriminationByMediumIsolation.Prediscriminants.preIso.Producer = \
        cms.InputTag("hpsPFTauDiscriminationByMediumChargedIsolation")

hpsPFTauDiscriminationByTightChargedIsolation = hpsPFTauDiscriminationByLooseChargedIsolation.clone()
hpsPFTauDiscriminationByTightChargedIsolation.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByTightChargedIsolation.qualityCuts.isolationQualityCuts.minGammaEt = 0.5
hpsPFTauDiscriminationByTightChargedIsolation.Prediscriminants.preIso = cms.PSet(
    Producer = cms.InputTag("hpsPFTauDiscriminationByMediumChargedIsolation"),
    cut = cms.double(0.5)
)

hpsPFTauDiscriminationByTightIsolation = hpsPFTauDiscriminationByLooseIsolation.clone()
hpsPFTauDiscriminationByTightIsolation.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByTightIsolation.qualityCuts.isolationQualityCuts.minGammaEt = 0.5
hpsPFTauDiscriminationByTightIsolation.maximumSumPtCut = \
        hpsPFTauDiscriminationByTightIsolation.qualityCuts.\
        isolationQualityCuts.minGammaEt
hpsPFTauDiscriminationByTightIsolation.Prediscriminants.preIso.Producer = \
        cms.InputTag("hpsPFTauDiscriminationByTightChargedIsolation")

#copying discriminator against electrons and muons
hpsPFTauDiscriminationByLooseElectronRejection = copy.deepcopy(pfRecoTauDiscriminationAgainstElectron)
setTauSource(hpsPFTauDiscriminationByLooseElectronRejection, 'hpsPFTauProducer')
hpsPFTauDiscriminationByLooseElectronRejection.Prediscriminants = noPrediscriminants
hpsPFTauDiscriminationByLooseElectronRejection.PFElectronMVA_maxValue = cms.double(0.6)

hpsPFTauDiscriminationByMediumElectronRejection = copy.deepcopy(pfRecoTauDiscriminationAgainstElectron)
setTauSource(hpsPFTauDiscriminationByMediumElectronRejection, 'hpsPFTauProducer')
hpsPFTauDiscriminationByMediumElectronRejection.Prediscriminants = noPrediscriminants
hpsPFTauDiscriminationByMediumElectronRejection.ApplyCut_EcalCrackCut = cms.bool(True)

hpsPFTauDiscriminationByTightElectronRejection = copy.deepcopy(pfRecoTauDiscriminationAgainstElectron)
setTauSource(hpsPFTauDiscriminationByTightElectronRejection, 'hpsPFTauProducer')
hpsPFTauDiscriminationByTightElectronRejection.Prediscriminants = noPrediscriminants
hpsPFTauDiscriminationByTightElectronRejection.ApplyCut_EcalCrackCut = cms.bool(True)
hpsPFTauDiscriminationByTightElectronRejection.ApplyCut_BremCombined = cms.bool(True)

hpsPFTauDiscriminationByLooseMuonRejection = copy.deepcopy(pfRecoTauDiscriminationAgainstMuon)
setTauSource(hpsPFTauDiscriminationByLooseMuonRejection, 'hpsPFTauProducer')
hpsPFTauDiscriminationByLooseMuonRejection.Prediscriminants = noPrediscriminants

hpsPFTauDiscriminationByTightMuonRejection = copy.deepcopy(pfRecoTauDiscriminationAgainstMuon)
setTauSource(hpsPFTauDiscriminationByTightMuonRejection, 'hpsPFTauProducer')
hpsPFTauDiscriminationByTightMuonRejection.Prediscriminants = noPrediscriminants
hpsPFTauDiscriminationByTightMuonRejection.discriminatorOption = cms.string('noAllArbitrated')


# Define the HPS selection discriminator used in cleaning
hpsSelectionDiscriminator.PFTauProducer = cms.InputTag("combinatoricRecoTaus")

# Define discriminants to use for HPS cleaning.
#hpsTightIsolationCleaner = hpsPFTauDiscriminationByTightIsolation.clone(
    #Prediscriminants = noPrediscriminants,
    #PFTauProducer = cms.InputTag("combinatoricRecoTaus"),
#)
#hpsMediumIsolationCleaner = hpsPFTauDiscriminationByMediumIsolation.clone(
    #Prediscriminants = noPrediscriminants,
    #PFTauProducer = cms.InputTag("combinatoricRecoTaus"),
#)
#hpsLooseIsolationCleaner = hpsPFTauDiscriminationByLooseIsolation.clone(
    #Prediscriminants = noPrediscriminants,
    #PFTauProducer = cms.InputTag("combinatoricRecoTaus"),
#)
#hpsVLooseIsolationCleaner = hpsPFTauDiscriminationByVLooseIsolation.clone(
    #Prediscriminants = noPrediscriminants,
    #PFTauProducer = cms.InputTag("combinatoricRecoTaus"),
#)

import RecoTauTag.RecoTau.RecoTauCleanerPlugins as cleaners

hpsPFTauProducer = cms.EDProducer(
    "RecoTauCleaner",
    src = cms.InputTag("combinatoricRecoTaus"),
    cleaners = cms.VPSet(
        # Prefer taus that dont' have charge == 3
        cleaners.unitCharge,
        # Prefer taus that are within DR<0.1 of the jet axis
        cleaners.matchingConeCut,
        # Prefer taus that pass HPS selections
        cms.PSet(
            name = cms.string("HPS_Select"),
            plugin = cms.string("RecoTauDiscriminantCleanerPlugin"),
            src = cms.InputTag("hpsSelectionDiscriminator"),
        ),
        cleaners.combinedIsolation
    )
)

produceHPSPFTaus = cms.Sequence(
    hpsSelectionDiscriminator
    #*hpsTightIsolationCleaner
    #*hpsMediumIsolationCleaner
    #*hpsLooseIsolationCleaner
    #*hpsVLooseIsolationCleaner
    *hpsPFTauProducer
)

produceAndDiscriminateHPSPFTaus = cms.Sequence(
    produceHPSPFTaus*
    hpsPFTauDiscriminationByDecayModeFinding*
    hpsPFTauDiscriminationByVLooseChargedIsolation*
    hpsPFTauDiscriminationByVLooseIsolation*
    hpsPFTauDiscriminationByLooseChargedIsolation*
    hpsPFTauDiscriminationByLooseIsolation*
    hpsPFTauDiscriminationByMediumChargedIsolation*
    hpsPFTauDiscriminationByMediumIsolation*
    hpsPFTauDiscriminationByTightChargedIsolation*
    hpsPFTauDiscriminationByTightIsolation*

    hpsPFTauDiscriminationByLooseElectronRejection*
    hpsPFTauDiscriminationByMediumElectronRejection*
    hpsPFTauDiscriminationByTightElectronRejection*
    hpsPFTauDiscriminationByLooseMuonRejection*
    hpsPFTauDiscriminationByTightMuonRejection
)
