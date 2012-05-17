import FWCore.ParameterSet.Config as cms
import copy

'''

Sequences for HPS taus

'''

# Define the discriminators for this tau
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi                      import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByMVAIsolation_cfi                   import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackFinding_cfi            import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectron_cfi                  import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectronMVA_cfi               import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectronMVA2_cfi              import *
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


hpsPFTauDiscriminationByVLooseIsolation = hpsPFTauDiscriminationByLooseIsolation.clone(
    customOuterCone = cms.double(0.3),
    isoConeSizeForDeltaBeta = cms.double(0.3),
    )
hpsPFTauDiscriminationByVLooseIsolation.qualityCuts.isolationQualityCuts.minTrackPt = 1.5
hpsPFTauDiscriminationByVLooseIsolation.qualityCuts.isolationQualityCuts.minGammaEt = 2.0
hpsPFTauDiscriminationByVLooseIsolation.Prediscriminants.preIso.Producer =  cms.InputTag("hpsPFTauDiscriminationByVLooseChargedIsolation")


hpsPFTauDiscriminationByMediumIsolation = hpsPFTauDiscriminationByLooseIsolation.clone()
hpsPFTauDiscriminationByMediumIsolation.qualityCuts.isolationQualityCuts.minTrackPt = 0.8
hpsPFTauDiscriminationByMediumIsolation.qualityCuts.isolationQualityCuts.minGammaEt = 0.8
hpsPFTauDiscriminationByMediumIsolation.Prediscriminants.preIso.Producer = cms.InputTag("hpsPFTauDiscriminationByMediumChargedIsolation")


hpsPFTauDiscriminationByTightIsolation = hpsPFTauDiscriminationByLooseIsolation.clone()
hpsPFTauDiscriminationByTightIsolation.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByTightIsolation.qualityCuts.isolationQualityCuts.minGammaEt = 0.5
hpsPFTauDiscriminationByTightIsolation.Prediscriminants.preIso.Producer = cms.InputTag("hpsPFTauDiscriminationByTightChargedIsolation")

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


hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr.clone(
    applySumPtCut = False,
    storeRawSumPt = cms.bool(True)
)

hpsPFTauDiscriminationByRawChargedIsolationDBSumPtCorr = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr.clone(
    applySumPtCut = False,
    ApplyDiscriminationByECALIsolation = False,
    storeRawSumPt = cms.bool(True)
)

hpsPFTauDiscriminationByRawGammaIsolationDBSumPtCorr = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr.clone(
    applySumPtCut = False,
    ApplyDiscriminationByTrackerIsolation = False,
    storeRawSumPt = cms.bool(True)
)


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

#Charge isolation based on combined isolation
hpsPFTauDiscriminationByVLooseChargedIsolation = hpsPFTauDiscriminationByVLooseCombinedIsolationDBSumPtCorr.clone(
        ApplyDiscriminationByECALIsolation = False
            )

hpsPFTauDiscriminationByLooseChargedIsolation = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr.clone(
            ApplyDiscriminationByECALIsolation = False
           )

hpsPFTauDiscriminationByMediumChargedIsolation = hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr.clone(
                ApplyDiscriminationByECALIsolation = False
                )
hpsPFTauDiscriminationByTightChargedIsolation = hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr.clone(
                ApplyDiscriminationByECALIsolation = False
                           )


hpsPFTauDiscriminationByChargedIsolationSeq = cms.Sequence(
        hpsPFTauDiscriminationByVLooseChargedIsolation*
            hpsPFTauDiscriminationByLooseChargedIsolation*
            hpsPFTauDiscriminationByMediumChargedIsolation*
            hpsPFTauDiscriminationByTightChargedIsolation
        )


# Define MVA based isolation discrimators
hpsPFTauDiscriminationByIsolationMVAraw = pfRecoTauDiscriminationByMVAIsolation.clone(
    PFTauProducer = cms.InputTag("hpsPFTauProducer"),
    Prediscriminants = requireDecayMode.clone(),
    returnMVA = cms.bool(True),
    )

hpsPFTauDiscriminationByLooseIsolationMVA = hpsPFTauDiscriminationByDecayModeFinding.clone(
    Prediscriminants = cms.PSet(
        BooleanOperator = cms.string("and"),
        mva = cms.PSet(
            Producer = cms.InputTag('hpsPFTauDiscriminationByIsolationMVAraw'),
            cut = cms.double(0.795)
        )
    ))
hpsPFTauDiscriminationByMediumIsolationMVA = copy.deepcopy(hpsPFTauDiscriminationByLooseIsolationMVA)
hpsPFTauDiscriminationByMediumIsolationMVA.Prediscriminants.mva.cut = cms.double(0.884)
hpsPFTauDiscriminationByTightIsolationMVA = copy.deepcopy(hpsPFTauDiscriminationByLooseIsolationMVA)
hpsPFTauDiscriminationByTightIsolationMVA.Prediscriminants.mva.cut = cms.double(0.921)

from RecoJets.Configuration.RecoPFJets_cff import kt6PFJets as dummy
kt6PFJetsForRhoComputationVoronoi = dummy.clone(
    doRhoFastjet = True,
    voronoiRfact = 0.9
)

hpsPFTauDiscriminationByMVAIsolationSeq = cms.Sequence(
    kt6PFJetsForRhoComputationVoronoi*
    hpsPFTauDiscriminationByIsolationMVAraw*
    hpsPFTauDiscriminationByLooseIsolationMVA*
    hpsPFTauDiscriminationByMediumIsolationMVA*
    hpsPFTauDiscriminationByTightIsolationMVA
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
    Prediscriminants = requireDecayMode.clone()
)

requireDecayMode = cms.PSet(
    BooleanOperator = cms.string("and"),
    decayMode = cms.PSet(
        Producer = cms.InputTag('hpsPFTauDiscriminationByDecayModeFinding'),
        cut = cms.double(0.5)
    )
)

from RecoTauTag.RecoTau.RecoTauDiscriminantCutMultiplexer_cfi import recoTauDiscriminantCutMultiplexer
hpsPFTauDiscriminationByMVA2Loose1ElectronRejection = recoTauDiscriminantCutMultiplexer.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByMVA2rawElectronRejection'),
    key = cms.InputTag('hpsPFTauDiscriminationByMVA2rawElectronRejection:category'),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0), # minMVA1prongNoEleMatchBL
            cut = cms.double(-0.99661791)
        ),
        cms.PSet(
            category = cms.uint32(1), # minMVA1prongBL
            cut = cms.double(-0.83979142)
        ),
        cms.PSet(
            category = cms.uint32(2), # minMVA1prongStripsWOgsfBL
            cut = cms.double(-0.99834895)
        ),
        cms.PSet(
            category = cms.uint32(3), # minMVA1prongStripsWgsfWOpfEleMvaBL
            cut = cms.double(-0.99925435)
        ),
        cms.PSet(
            category = cms.uint32(4), # minMVA1prongStripsWgsfWpfEleMvaBL
            cut = cms.double(-0.87972307)
        ),
        cms.PSet(
            category = cms.uint32(5), # minMVA1prongNoEleMatchEC
            cut = cms.double(-0.99465221)
        ),
        cms.PSet(
            category = cms.uint32(6), # minMVA1prongEC
            cut = cms.double(-0.7369861)
        ),
        cms.PSet(
            category = cms.uint32(7), # minMVA1prongStripsWOgsfEC
            cut = cms.double(-0.99759883)
        ),
        cms.PSet(
            category = cms.uint32(8), # minMVA1prongStripsWgsfWOpfEleMvaEC
            cut = cms.double(-0.99703252)
        ),
        cms.PSet(
            category = cms.uint32(9), # minMVA1prongStripsWgsfWpfEleMvaEC
            cut = cms.double(-0.70050365)
        )
    )
)

hpsPFTauDiscriminationByMVA2Loose2ElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA2Loose1ElectronRejection)
hpsPFTauDiscriminationByMVA2Loose2ElectronRejection.Prediscriminants = requireDecayMode.clone(
    decayMode = cms.PSet(
        Producer = cms.InputTag('hpsPFTauDiscriminationByMVA2Loose1ElectronRejection'),
        cut = cms.double(0.5)
    )
)
hpsPFTauDiscriminationByMVA2Loose2ElectronRejection.mapping[0].cut = cms.double(-0.99661791)
hpsPFTauDiscriminationByMVA2Loose2ElectronRejection.mapping[1].cut = cms.double(-0.67843455)
hpsPFTauDiscriminationByMVA2Loose2ElectronRejection.mapping[2].cut = cms.double(-0.98259228)
hpsPFTauDiscriminationByMVA2Loose2ElectronRejection.mapping[3].cut = cms.double(-0.99925435)
hpsPFTauDiscriminationByMVA2Loose2ElectronRejection.mapping[4].cut = cms.double(-0.63197774)
hpsPFTauDiscriminationByMVA2Loose2ElectronRejection.mapping[5].cut = cms.double(-0.99465221)
hpsPFTauDiscriminationByMVA2Loose2ElectronRejection.mapping[6].cut = cms.double(-0.52862346)
hpsPFTauDiscriminationByMVA2Loose2ElectronRejection.mapping[7].cut = cms.double(-0.89986807)
hpsPFTauDiscriminationByMVA2Loose2ElectronRejection.mapping[8].cut = cms.double(-0.82358503)
hpsPFTauDiscriminationByMVA2Loose2ElectronRejection.mapping[9].cut = cms.double(-0.37769419)

hpsPFTauDiscriminationByMVA2Medium1ElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA2Loose1ElectronRejection)
hpsPFTauDiscriminationByMVA2Medium1ElectronRejection.Prediscriminants = requireDecayMode.clone(
    decayMode = cms.PSet(
        Producer = cms.InputTag('hpsPFTauDiscriminationByMVA2Loose2ElectronRejection'),
        cut = cms.double(0.5)
    )
)
hpsPFTauDiscriminationByMVA2Medium1ElectronRejection.mapping[0].cut = cms.double(-0.99661791)
hpsPFTauDiscriminationByMVA2Medium1ElectronRejection.mapping[1].cut = cms.double(-0.26334548)
hpsPFTauDiscriminationByMVA2Medium1ElectronRejection.mapping[2].cut = cms.double(-0.75781053)
hpsPFTauDiscriminationByMVA2Medium1ElectronRejection.mapping[3].cut = cms.double(-0.99925435)
hpsPFTauDiscriminationByMVA2Medium1ElectronRejection.mapping[4].cut = cms.double(-0.032793567)
hpsPFTauDiscriminationByMVA2Medium1ElectronRejection.mapping[5].cut = cms.double(-0.99465221)
hpsPFTauDiscriminationByMVA2Medium1ElectronRejection.mapping[6].cut = cms.double(-0.10370751)
hpsPFTauDiscriminationByMVA2Medium1ElectronRejection.mapping[7].cut = cms.double(-0.31169191)
hpsPFTauDiscriminationByMVA2Medium1ElectronRejection.mapping[8].cut = cms.double(-0.24109723)
hpsPFTauDiscriminationByMVA2Medium1ElectronRejection.mapping[9].cut = cms.double(+0.19629943)

hpsPFTauDiscriminationByMVA2Medium2ElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA2Loose1ElectronRejection)
hpsPFTauDiscriminationByMVA2Medium2ElectronRejection.Prediscriminants = requireDecayMode.clone(
    decayMode = cms.PSet(
        Producer = cms.InputTag('hpsPFTauDiscriminationByMVA2Medium1ElectronRejection'),
        cut = cms.double(0.5)
    )
)
hpsPFTauDiscriminationByMVA2Medium2ElectronRejection.mapping[0].cut = cms.double(-0.99661791)
hpsPFTauDiscriminationByMVA2Medium2ElectronRejection.mapping[1].cut = cms.double(+0.62861615)
hpsPFTauDiscriminationByMVA2Medium2ElectronRejection.mapping[2].cut = cms.double(+0.54859632)
hpsPFTauDiscriminationByMVA2Medium2ElectronRejection.mapping[3].cut = cms.double(-0.33699143)
hpsPFTauDiscriminationByMVA2Medium2ElectronRejection.mapping[4].cut = cms.double(+0.67368031)
hpsPFTauDiscriminationByMVA2Medium2ElectronRejection.mapping[5].cut = cms.double(-0.42534944)
hpsPFTauDiscriminationByMVA2Medium2ElectronRejection.mapping[6].cut = cms.double(+0.71575898)
hpsPFTauDiscriminationByMVA2Medium2ElectronRejection.mapping[7].cut = cms.double(+0.77528948)
hpsPFTauDiscriminationByMVA2Medium2ElectronRejection.mapping[8].cut = cms.double(+0.6177476)
hpsPFTauDiscriminationByMVA2Medium2ElectronRejection.mapping[9].cut = cms.double(+0.75739658)

hpsPFTauDiscriminationByMVA2Tight1ElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA2Loose1ElectronRejection)
hpsPFTauDiscriminationByMVA2Tight1ElectronRejection.Prediscriminants = requireDecayMode.clone(
    decayMode = cms.PSet(
        Producer = cms.InputTag('hpsPFTauDiscriminationByMVA2Medium2ElectronRejection'),
        cut = cms.double(0.5)
    )
)
hpsPFTauDiscriminationByMVA2Tight1ElectronRejection.mapping[0].cut = cms.double(+0.18199094)
hpsPFTauDiscriminationByMVA2Tight1ElectronRejection.mapping[1].cut = cms.double(+0.99951702)
hpsPFTauDiscriminationByMVA2Tight1ElectronRejection.mapping[2].cut = cms.double(+0.97502345)
hpsPFTauDiscriminationByMVA2Tight1ElectronRejection.mapping[3].cut = cms.double(+0.85220331)
hpsPFTauDiscriminationByMVA2Tight1ElectronRejection.mapping[4].cut = cms.double(+0.96358234)
hpsPFTauDiscriminationByMVA2Tight1ElectronRejection.mapping[5].cut = cms.double(+0.70447916)
hpsPFTauDiscriminationByMVA2Tight1ElectronRejection.mapping[6].cut = cms.double(+0.96727246)
hpsPFTauDiscriminationByMVA2Tight1ElectronRejection.mapping[7].cut = cms.double(+0.98667461)
hpsPFTauDiscriminationByMVA2Tight1ElectronRejection.mapping[8].cut = cms.double(+0.93107212)
hpsPFTauDiscriminationByMVA2Tight1ElectronRejection.mapping[9].cut = cms.double(+0.91691643)

hpsPFTauDiscriminationByMVA2Tight2ElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA2Loose1ElectronRejection)
hpsPFTauDiscriminationByMVA2Tight2ElectronRejection.Prediscriminants = requireDecayMode.clone(
    decayMode = cms.PSet(
        Producer = cms.InputTag('hpsPFTauDiscriminationByMVA2Tight1ElectronRejection'),
        cut = cms.double(0.5)
    )
)
hpsPFTauDiscriminationByMVA2Tight2ElectronRejection.mapping[0].cut = cms.double(+0.88839751)
hpsPFTauDiscriminationByMVA2Tight2ElectronRejection.mapping[1].cut = cms.double(+0.99951702)
hpsPFTauDiscriminationByMVA2Tight2ElectronRejection.mapping[2].cut = cms.double(+0.9901818)
hpsPFTauDiscriminationByMVA2Tight2ElectronRejection.mapping[3].cut = cms.double(+0.94295716)
hpsPFTauDiscriminationByMVA2Tight2ElectronRejection.mapping[4].cut = cms.double(+0.98096448)
hpsPFTauDiscriminationByMVA2Tight2ElectronRejection.mapping[5].cut = cms.double(+0.89816976)
hpsPFTauDiscriminationByMVA2Tight2ElectronRejection.mapping[6].cut = cms.double(+0.99304312)
hpsPFTauDiscriminationByMVA2Tight2ElectronRejection.mapping[7].cut = cms.double(+0.98667461)
hpsPFTauDiscriminationByMVA2Tight2ElectronRejection.mapping[8].cut = cms.double(+0.96963817)
hpsPFTauDiscriminationByMVA2Tight2ElectronRejection.mapping[9].cut = cms.double(+0.91691643)

hpsPFTauDiscriminationByMVA2VTight1ElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA2Loose1ElectronRejection)
hpsPFTauDiscriminationByMVA2VTight1ElectronRejection.Prediscriminants = requireDecayMode.clone(
    decayMode = cms.PSet(
        Producer = cms.InputTag('hpsPFTauDiscriminationByMVA2Tight2ElectronRejection'),
        cut = cms.double(0.5)
    )
)
hpsPFTauDiscriminationByMVA2VTight1ElectronRejection.mapping[0].cut = cms.double(+0.94070917)
hpsPFTauDiscriminationByMVA2VTight1ElectronRejection.mapping[1].cut = cms.double(+0.99951702)
hpsPFTauDiscriminationByMVA2VTight1ElectronRejection.mapping[2].cut = cms.double(+0.9901818)
hpsPFTauDiscriminationByMVA2VTight1ElectronRejection.mapping[3].cut = cms.double(+0.96234727)
hpsPFTauDiscriminationByMVA2VTight1ElectronRejection.mapping[4].cut = cms.double(+0.98695832)
hpsPFTauDiscriminationByMVA2VTight1ElectronRejection.mapping[5].cut = cms.double(+0.95482075)
hpsPFTauDiscriminationByMVA2VTight1ElectronRejection.mapping[6].cut = cms.double(+0.99424177)
hpsPFTauDiscriminationByMVA2VTight1ElectronRejection.mapping[7].cut = cms.double(+0.98667461)
hpsPFTauDiscriminationByMVA2VTight1ElectronRejection.mapping[8].cut = cms.double(+0.97783101)
hpsPFTauDiscriminationByMVA2VTight1ElectronRejection.mapping[9].cut = cms.double(+0.91691643)

hpsPFTauDiscriminationByMVA2VTight2ElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA2Loose1ElectronRejection)
hpsPFTauDiscriminationByMVA2VTight2ElectronRejection.Prediscriminants = requireDecayMode.clone(
    decayMode = cms.PSet(
        Producer = cms.InputTag('hpsPFTauDiscriminationByMVA2VTight1ElectronRejection'),
        cut = cms.double(0.5)
    )
)
hpsPFTauDiscriminationByMVA2VTight2ElectronRejection.mapping[0].cut = cms.double(+0.95768046)
hpsPFTauDiscriminationByMVA2VTight2ElectronRejection.mapping[1].cut = cms.double(+0.99951702)
hpsPFTauDiscriminationByMVA2VTight2ElectronRejection.mapping[2].cut = cms.double(+0.99097961)
hpsPFTauDiscriminationByMVA2VTight2ElectronRejection.mapping[3].cut = cms.double(+0.97214228)
hpsPFTauDiscriminationByMVA2VTight2ElectronRejection.mapping[4].cut = cms.double(+0.98955566)
hpsPFTauDiscriminationByMVA2VTight2ElectronRejection.mapping[5].cut = cms.double(+0.97397041)
hpsPFTauDiscriminationByMVA2VTight2ElectronRejection.mapping[6].cut = cms.double(+0.99424177)
hpsPFTauDiscriminationByMVA2VTight2ElectronRejection.mapping[7].cut = cms.double(+0.98667461)
hpsPFTauDiscriminationByMVA2VTight2ElectronRejection.mapping[8].cut = cms.double(+0.98542428)
hpsPFTauDiscriminationByMVA2VTight2ElectronRejection.mapping[9].cut = cms.double(+0.91691643)

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
    hpsPFTauDiscriminationByMVAIsolationSeq*

    hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByRawChargedIsolationDBSumPtCorr*
    hpsPFTauDiscriminationByRawGammaIsolationDBSumPtCorr*

    hpsPFTauDiscriminationByCombinedIsolationSeqDBSumPtCorr*
    hpsPFTauDiscriminationByLooseElectronRejection*
    hpsPFTauDiscriminationByMediumElectronRejection*
    hpsPFTauDiscriminationByTightElectronRejection*
    hpsPFTauDiscriminationByMVAElectronRejection*
    hpsPFTauDiscriminationByMVA2rawElectronRejection*
    hpsPFTauDiscriminationByMVA2Loose1ElectronRejection*
    hpsPFTauDiscriminationByMVA2Loose2ElectronRejection*
    hpsPFTauDiscriminationByMVA2Medium1ElectronRejection*
    hpsPFTauDiscriminationByMVA2Medium2ElectronRejection*
    hpsPFTauDiscriminationByMVA2Tight1ElectronRejection*
    hpsPFTauDiscriminationByMVA2Tight2ElectronRejection*
    hpsPFTauDiscriminationByMVA2VTight1ElectronRejection*
    hpsPFTauDiscriminationByMVA2VTight2ElectronRejection*
    hpsPFTauDiscriminationByLooseMuonRejection*
    hpsPFTauDiscriminationByMediumMuonRejection*
    hpsPFTauDiscriminationByTightMuonRejection
)
