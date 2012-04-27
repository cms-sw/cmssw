import FWCore.ParameterSet.Config as cms
import copy

'''

Sequences for HPS taus

'''

# Define the discriminators for this tau
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi                      import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackFinding_cfi            import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectron_cfi                  import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectronMVA_cfi               import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectronMVA2_cfi               import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuon_cfi                      import *

# Load helper functions to change the source of the discriminants
from RecoTauTag.RecoTau.TauDiscriminatorTools import *

# Select those taus that pass the HPS selections
#  - pt > 15, mass cuts, tauCone cut
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByHPSSelection_cfi import hpsSelectionDiscriminator
hpsPFTauDiscriminationByDecayModeFinding = hpsSelectionDiscriminator.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer')
    )

# Define decay mode prediscriminant
requireDecayMode = cms.PSet(
    BooleanOperator = cms.string("and"),
    decayMode = cms.PSet(
        Producer = cms.InputTag('hpsPFTauDiscriminationByDecayModeFinding'),
        cut = cms.double(0.5)
    )
)

# First apply only charged isolation
hpsPFTauDiscriminationByLooseChargedIsolation = pfRecoTauDiscriminationByIsolation.clone(
    PFTauProducer = cms.InputTag("hpsPFTauProducer"),
    Prediscriminants = requireDecayMode.clone(),
    ApplyDiscriminationByECALIsolation = False
)

#Building the prototype for  the Discriminator by Isolation
hpsPFTauDiscriminationByLooseIsolation = pfRecoTauDiscriminationByIsolation.clone(
    PFTauProducer = cms.InputTag("hpsPFTauProducer"),
    Prediscriminants = requireDecayMode.clone(),
    ApplyDiscriminationByTrackerIsolation = False,
    ApplyDiscriminationByECALIsolation = True,
    applyOccupancyCut = True
)
hpsPFTauDiscriminationByLooseIsolation.Prediscriminants.preIso = cms.PSet(
    Producer = cms.InputTag("hpsPFTauDiscriminationByLooseChargedIsolation"),
    cut = cms.double(0.5))


# Make an even looser discriminator
hpsPFTauDiscriminationByVLooseChargedIsolation = hpsPFTauDiscriminationByLooseChargedIsolation.clone(
    customOuterCone = cms.double(0.3),
    isoConeSizeForDeltaBeta = cms.double(0.3)
    )
hpsPFTauDiscriminationByVLooseChargedIsolation.qualityCuts.isolationQualityCuts.minTrackPt = 1.5
hpsPFTauDiscriminationByVLooseChargedIsolation.qualityCuts.isolationQualityCuts.minGammaEt = 2.0

hpsPFTauDiscriminationByVLooseIsolation = hpsPFTauDiscriminationByLooseIsolation.clone(
    customOuterCone = cms.double(0.3),
    isoConeSizeForDeltaBeta = cms.double(0.3),
    )
hpsPFTauDiscriminationByVLooseIsolation.qualityCuts.isolationQualityCuts.minTrackPt = 1.5
hpsPFTauDiscriminationByVLooseIsolation.qualityCuts.isolationQualityCuts.minGammaEt = 2.0
hpsPFTauDiscriminationByVLooseIsolation.Prediscriminants.preIso.Producer =  cms.InputTag("hpsPFTauDiscriminationByVLooseChargedIsolation")


hpsPFTauDiscriminationByMediumChargedIsolation = hpsPFTauDiscriminationByLooseChargedIsolation.clone()
hpsPFTauDiscriminationByMediumChargedIsolation.qualityCuts.isolationQualityCuts.minTrackPt = 0.8
hpsPFTauDiscriminationByMediumChargedIsolation.qualityCuts.isolationQualityCuts.minGammaEt = 0.8
hpsPFTauDiscriminationByMediumChargedIsolation.Prediscriminants.preIso = cms.PSet(
    Producer = cms.InputTag("hpsPFTauDiscriminationByLooseChargedIsolation"),
    cut = cms.double(0.5))

hpsPFTauDiscriminationByMediumIsolation = hpsPFTauDiscriminationByLooseIsolation.clone()
hpsPFTauDiscriminationByMediumIsolation.qualityCuts.isolationQualityCuts.minTrackPt = 0.8
hpsPFTauDiscriminationByMediumIsolation.qualityCuts.isolationQualityCuts.minGammaEt = 0.8
hpsPFTauDiscriminationByMediumIsolation.Prediscriminants.preIso.Producer = cms.InputTag("hpsPFTauDiscriminationByMediumChargedIsolation")


hpsPFTauDiscriminationByTightChargedIsolation = hpsPFTauDiscriminationByLooseChargedIsolation.clone()
hpsPFTauDiscriminationByTightChargedIsolation.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByTightChargedIsolation.qualityCuts.isolationQualityCuts.minGammaEt = 0.5
hpsPFTauDiscriminationByTightChargedIsolation.Prediscriminants.preIso = cms.PSet(
    Producer = cms.InputTag("hpsPFTauDiscriminationByMediumChargedIsolation"),
    cut = cms.double(0.5))

hpsPFTauDiscriminationByTightIsolation = hpsPFTauDiscriminationByLooseIsolation.clone()
hpsPFTauDiscriminationByTightIsolation.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByTightIsolation.qualityCuts.isolationQualityCuts.minGammaEt = 0.5
hpsPFTauDiscriminationByTightIsolation.Prediscriminants.preIso.Producer = cms.InputTag("hpsPFTauDiscriminationByTightChargedIsolation")

hpsPFTauDiscriminationByChargedIsolationSeq = cms.Sequence(
    hpsPFTauDiscriminationByVLooseChargedIsolation*
    hpsPFTauDiscriminationByLooseChargedIsolation*
    hpsPFTauDiscriminationByMediumChargedIsolation*
    hpsPFTauDiscriminationByTightChargedIsolation
)

hpsPFTauDiscriminationByIsolationSeq = cms.Sequence(
    hpsPFTauDiscriminationByVLooseIsolation*
    hpsPFTauDiscriminationByLooseIsolation*
    hpsPFTauDiscriminationByMediumIsolation*
    hpsPFTauDiscriminationByTightIsolation
)

_isolation_types = ['VLoose', 'Loose', 'Medium', 'Tight']
# Now build the sequences that apply PU corrections

# Make Delta Beta corrections (on SumPt quantity)
hpsPFTauDiscriminationByVLooseIsolationDBSumPtCorr = hpsPFTauDiscriminationByVLooseIsolation.clone(
    deltaBetaPUTrackPtCutOverride = cms.double(0.5),
    applyDeltaBetaCorrection = True,
    isoConeSizeForDeltaBeta = 0.8,
    deltaBetaFactor = "%0.4f"%(0.0123/0.1687),
    applyOccupancyCut = False,
    applySumPtCut = True,
    )

hpsPFTauDiscriminationByVLooseIsolationDBSumPtCorr.maximumSumPtCut=hpsPFTauDiscriminationByVLooseIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt

hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr = hpsPFTauDiscriminationByLooseIsolation.clone(
    deltaBetaPUTrackPtCutOverride = cms.double(0.5),
    applyDeltaBetaCorrection = True,
    isoConeSizeForDeltaBeta = 0.8,
    deltaBetaFactor = "%0.4f"%(0.0123/0.1687),
    applyOccupancyCut = False,
    applySumPtCut = True,
    )
hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr.maximumSumPtCut=hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt


hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr = hpsPFTauDiscriminationByMediumIsolation.clone(
    deltaBetaPUTrackPtCutOverride = cms.double(0.5),
    applyDeltaBetaCorrection = True,
    isoConeSizeForDeltaBeta = 0.8,
    deltaBetaFactor = "%0.4f"%(0.0462/0.1687),
    applyOccupancyCut = False,
    applySumPtCut = True,
    )
hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr.maximumSumPtCut=hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt


hpsPFTauDiscriminationByTightIsolationDBSumPtCorr = hpsPFTauDiscriminationByTightIsolation.clone(
    deltaBetaPUTrackPtCutOverride = cms.double(0.5),
    applyDeltaBetaCorrection = True,
    isoConeSizeForDeltaBeta = 0.8,
    deltaBetaFactor = "%0.4f"%(0.0772/0.1687),
    applyOccupancyCut = False,
    applySumPtCut = True,
    )
hpsPFTauDiscriminationByTightIsolationDBSumPtCorr.maximumSumPtCut=hpsPFTauDiscriminationByTightIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt

hpsPFTauDiscriminationByIsolationSeqDBSumPtCorr = cms.Sequence(
    hpsPFTauDiscriminationByVLooseIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByTightIsolationDBSumPtCorr
    )


hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr = hpsPFTauDiscriminationByVLooseIsolationDBSumPtCorr.clone(
    ApplyDiscriminationByTrackerIsolation = True,
    ApplyDiscriminationByECALIsolation = True,
    deltaBetaFactor = "%0.4f"%(0.0772/0.1687),
    applyOccupancyCut = False,
    applySumPtCut = True,
    maximumSumPtCut = 3.0,
    Prediscriminants = requireDecayMode.clone()
    )
hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt = 0.5

hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr = hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr.clone(
    applySumPtCut = False,
    storeRawSumPt = cms.bool(True)
)

hpsPFTauDiscriminationByRawChargedIsolationDBSumPtCorr = hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr.clone(
    applySumPtCut = False,
    ApplyDiscriminationByECALIsolation = False,
    storeRawSumPt = cms.bool(True)
)

hpsPFTauDiscriminationByRawGammaIsolationDBSumPtCorr = hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr.clone(
    applySumPtCut = False,
    ApplyDiscriminationByTrackerIsolation = False,
    storeRawSumPt = cms.bool(True)
)

hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr = hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr.clone(
    ApplyDiscriminationByTrackerIsolation = True,
    ApplyDiscriminationByECALIsolation = True,
    deltaBetaFactor = "%0.4f"%(0.0772/0.1687),
    applyOccupancyCut = False,
    applySumPtCut = True,
    maximumSumPtCut = 2.0,
    Prediscriminants = requireDecayMode.clone()
    )
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt = 0.5

hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr = hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr.clone(
    ApplyDiscriminationByTrackerIsolation = True,
    ApplyDiscriminationByECALIsolation = True,
    deltaBetaFactor = "%0.4f"%(0.0772/0.1687),
    applyOccupancyCut = False,
    applySumPtCut = True,
    maximumSumPtCut = 1.0,
    Prediscriminants = requireDecayMode.clone()
    )
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt = 0.5

hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr = hpsPFTauDiscriminationByTightIsolationDBSumPtCorr.clone(
    ApplyDiscriminationByTrackerIsolation = True,
    ApplyDiscriminationByECALIsolation = True,
    deltaBetaFactor = "%0.4f"%(0.0772/0.1687),
    applyOccupancyCut = False,
    applySumPtCut = True,
    maximumSumPtCut = 0.8,
    Prediscriminants = requireDecayMode.clone()
    )
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt = 0.5

hpsPFTauDiscriminationByCombinedIsolationSeqDBSumPtCorr = cms.Sequence(
    hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr
    )

#copying discriminator against electrons and muons
hpsPFTauDiscriminationByLooseElectronRejection = pfRecoTauDiscriminationAgainstElectron.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = noPrediscriminants,
    PFElectronMVA_maxValue = cms.double(0.6)
    )

hpsPFTauDiscriminationByMediumElectronRejection = pfRecoTauDiscriminationAgainstElectron.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = noPrediscriminants,
    ApplyCut_EcalCrackCut = cms.bool(True)
    )

hpsPFTauDiscriminationByTightElectronRejection = pfRecoTauDiscriminationAgainstElectron.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = noPrediscriminants,
    ApplyCut_EcalCrackCut = cms.bool(True),
    ApplyCut_BremCombined = cms.bool(True)
    )

hpsPFTauDiscriminationByLooseMuonRejection = pfRecoTauDiscriminationAgainstMuon.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = noPrediscriminants
    )

hpsPFTauDiscriminationByMediumMuonRejection = pfRecoTauDiscriminationAgainstMuon.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = noPrediscriminants,
    discriminatorOption = cms.string('noAllArbitrated')
    )

hpsPFTauDiscriminationByTightMuonRejection = pfRecoTauDiscriminationAgainstMuon.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = noPrediscriminants,
    discriminatorOption = cms.string('noAllArbitratedWithHOP')
    )

hpsPFTauDiscriminationByMVAElectronRejection = pfRecoTauDiscriminationAgainstElectronMVA.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
)

# Additionally require that the MVA electrons pass electron medium
# (this discriminator was used on the training sample)
hpsPFTauDiscriminationByMVAElectronRejection.Prediscriminants.electronMedium = \
        cms.PSet(
            Producer = cms.InputTag('hpsPFTauDiscriminationByMediumElectronRejection'),
            cut = cms.double(0.5)
        )

hpsPFTauDiscriminationByMVA2rawElectronRejection = pfRecoTauDiscriminationAgainstElectronMVA2.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
)

# Additionally require that the MVA2 electrons pass electron loose
# (this discriminator was used on the training sample)
hpsPFTauDiscriminationByMVA2rawElectronRejection.Prediscriminants.electronLoose = \
        cms.PSet(
            Producer = cms.InputTag('hpsPFTauDiscriminationByLooseElectronRejection'),
            cut = cms.double(0.5)
        )

hpsPFTauDiscriminationByMVA2LooseElectronRejection = hpsPFTauDiscriminationByMVA2rawElectronRejection.clone(
    returnMVA = cms.bool(False),
    # define 95% signal efficiency WP
    minMVA1prongNoEleMatchBL           = cms.double(-0.101727),
    minMVA1prongBL                     = cms.double(-0.130411),
    minMVA1prongStripsWOgsfBL          = cms.double(-0.110745),
    minMVA1prongStripsWgsfWOpfEleMvaBL = cms.double(-0.11647),
    minMVA1prongStripsWgsfWpfEleMvaBL  = cms.double(-0.134414),
    minMVA1prongNoEleMatchEC           = cms.double(-0.169389),
    minMVA1prongEC                     = cms.double(-0.140327),
    minMVA1prongStripsWOgsfEC          = cms.double(-0.102089),
    minMVA1prongStripsWgsfWOpfEleMvaEC = cms.double(-0.14057),
    minMVA1prongStripsWgsfWpfEleMvaEC  = cms.double(-0.0975809)
)

hpsPFTauDiscriminationByMVA2MediumElectronRejection = hpsPFTauDiscriminationByMVA2rawElectronRejection.clone(
    returnMVA = cms.bool(False),
    # define 85% signal efficiency WP
    minMVA1prongNoEleMatchBL           = cms.double(-0.0727222),
    minMVA1prongBL                     = cms.double(-0.072778),
    minMVA1prongStripsWOgsfBL          = cms.double(-0.137213),
    minMVA1prongStripsWgsfWOpfEleMvaBL = cms.double(-0.0948499),
    minMVA1prongStripsWgsfWpfEleMvaBL  = cms.double(-0.0600284),
    minMVA1prongNoEleMatchEC           = cms.double(-0.084118),
    minMVA1prongEC                     = cms.double(+0.0648186),
    minMVA1prongStripsWOgsfEC          = cms.double(-0.0804441),
    minMVA1prongStripsWgsfWOpfEleMvaEC = cms.double(-0.0227585),
    minMVA1prongStripsWgsfWpfEleMvaEC  = cms.double(-0.116097)
)

hpsPFTauDiscriminationByMVA2TightElectronRejection = hpsPFTauDiscriminationByMVA2rawElectronRejection.clone(
    returnMVA = cms.bool(False),
    # define 75% signal efficiency WP
    minMVA1prongNoEleMatchBL           = cms.double(-0.126102),
    minMVA1prongBL                     = cms.double(-0.0458154),
    minMVA1prongStripsWOgsfBL          = cms.double(-0.137043),
    minMVA1prongStripsWgsfWOpfEleMvaBL = cms.double(+0.0332071),
    minMVA1prongStripsWgsfWpfEleMvaBL  = cms.double(-0.0448832),
    minMVA1prongNoEleMatchEC           = cms.double(-0.0177012),
    minMVA1prongEC                     = cms.double(+0.189192),
    minMVA1prongStripsWOgsfEC          = cms.double(-0.0444424),
    minMVA1prongStripsWgsfWOpfEleMvaEC = cms.double(-0.0938333),
    minMVA1prongStripsWgsfWpfEleMvaEC  = cms.double(+0.144127)
)

hpsPFTauDiscriminationByMVA2VTightElectronRejection = hpsPFTauDiscriminationByMVA2rawElectronRejection.clone(
    returnMVA = cms.bool(False),
    # define 70% signal efficiency WP
    minMVA1prongNoEleMatchBL           = cms.double(-0.133868),
    minMVA1prongBL                     = cms.double(-0.0295349),
    minMVA1prongStripsWOgsfBL          = cms.double(-0.00930688),
    minMVA1prongStripsWgsfWOpfEleMvaBL = cms.double(-0.0949625),
    minMVA1prongStripsWgsfWpfEleMvaBL  = cms.double(+0.832122),
    minMVA1prongNoEleMatchEC           = cms.double(-0.00877876),
    minMVA1prongEC                     = cms.double(+0.275851),
    minMVA1prongStripsWOgsfEC          = cms.double(-0.0145072),
    minMVA1prongStripsWgsfWOpfEleMvaEC = cms.double(+0.515682),
    minMVA1prongStripsWgsfWpfEleMvaEC  = cms.double(-0.0634265)
)

# Define the HPS selection discriminator used in cleaning
hpsSelectionDiscriminator.PFTauProducer = cms.InputTag("combinatoricRecoTaus")

import RecoTauTag.RecoTau.RecoTauCleanerPlugins as cleaners

hpsPFTauProducerSansRefs = cms.EDProducer(
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

hpsPFTauProducer = cms.EDProducer(
    "RecoTauPiZeroUnembedder",
    src = cms.InputTag("hpsPFTauProducerSansRefs")
)

produceHPSPFTaus = cms.Sequence(
    hpsSelectionDiscriminator
    #*hpsTightIsolationCleaner
    #*hpsMediumIsolationCleaner
    #*hpsLooseIsolationCleaner
    #*hpsVLooseIsolationCleaner
    *hpsPFTauProducerSansRefs
    *hpsPFTauProducer
)

produceAndDiscriminateHPSPFTaus = cms.Sequence(
    produceHPSPFTaus*
    hpsPFTauDiscriminationByDecayModeFinding*
    hpsPFTauDiscriminationByChargedIsolationSeq*
    hpsPFTauDiscriminationByIsolationSeq*
    #hpsPFTauDiscriminationByIsolationSeqRhoCorr*
    #hpsPFTauDiscriminationByIsolationSeqCustomRhoCorr*
    hpsPFTauDiscriminationByIsolationSeqDBSumPtCorr*

    hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByRawChargedIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByRawGammaIsolationDBSumPtCorr*

    hpsPFTauDiscriminationByCombinedIsolationSeqDBSumPtCorr*
    hpsPFTauDiscriminationByLooseElectronRejection*
    hpsPFTauDiscriminationByMediumElectronRejection*
    hpsPFTauDiscriminationByTightElectronRejection*
    hpsPFTauDiscriminationByMVAElectronRejection*
    hpsPFTauDiscriminationByMVA2rawElectronRejection*
    hpsPFTauDiscriminationByMVA2LooseElectronRejection*
    hpsPFTauDiscriminationByMVA2MediumElectronRejection*
    hpsPFTauDiscriminationByMVA2TightElectronRejection*
    hpsPFTauDiscriminationByMVA2VTightElectronRejection*
    hpsPFTauDiscriminationByLooseMuonRejection*
    hpsPFTauDiscriminationByMediumMuonRejection*
    hpsPFTauDiscriminationByTightMuonRejection
)
