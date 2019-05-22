import FWCore.ParameterSet.Config as cms
import copy
from Configuration.Eras.Modifier_phase2_common_cff import phase2_common

'''

Sequences for HPS taus

'''

## Discriminator sources
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi                      import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByLeadingTrackFinding_cfi            import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectron_cfi                  import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectronMVA6_cfi              import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstElectronDeadECAL_cfi          import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuon_cfi                      import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuon2_cfi                     import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuonMVA_cfi                   import *

from RecoTauTag.RecoTau.RecoTauDiscriminantCutMultiplexer_cfi import *
## Helper functions to change the source of the discriminants
from RecoTauTag.RecoTau.TauDiscriminatorTools import *
## PFjet input parameters
from RecoTauTag.RecoTau.PFRecoTauPFJetInputs_cfi import PFRecoTauPFJetInputs
## DeltaBeta correction factor
ak4dBetaCorrection = 0.20
## MVAs from SQLlite file/prep. DB
from RecoTauTag.Configuration.loadRecoTauTagMVAsFromPrepDB_cfi import *

## Selection of taus that pass the HPS selections: pt > 15, mass cuts, tauCone cut
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByHPSSelection_cfi import hpsSelectionDiscriminator, decayMode_1Prong0Pi0, decayMode_1Prong1Pi0, decayMode_1Prong2Pi0, decayMode_2Prong0Pi0, decayMode_2Prong1Pi0, decayMode_3Prong0Pi0, decayMode_3Prong1Pi0

hpsPFTauDiscriminationByDecayModeFindingNewDMs = hpsSelectionDiscriminator.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    #----------------------------------------------------------------------------
    # CV: disable 3Prong1Pi0 decay mode
    decayModes = cms.VPSet(
        decayMode_1Prong0Pi0,
        decayMode_1Prong1Pi0,
        decayMode_1Prong2Pi0,
        decayMode_2Prong0Pi0,
        decayMode_2Prong1Pi0,
        decayMode_3Prong0Pi0,
        decayMode_3Prong1Pi0,
    )
    #----------------------------------------------------------------------------
)
hpsPFTauDiscriminationByDecayModeFindingOldDMs = hpsSelectionDiscriminator.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    decayModes = cms.VPSet(
        decayMode_1Prong0Pi0,
        decayMode_1Prong1Pi0,
        decayMode_1Prong2Pi0,
        decayMode_3Prong0Pi0
    ),
    requireTauChargedHadronsToBeChargedPFCands = cms.bool(True)
)
hpsPFTauDiscriminationByDecayModeFinding = hpsPFTauDiscriminationByDecayModeFindingOldDMs.clone() ## CV: kept for backwards compatibility

## Decay mode prediscriminant
requireDecayMode = cms.PSet(
    BooleanOperator = cms.string("and"),
    decayMode = cms.PSet(
        Producer = cms.InputTag('hpsPFTauDiscriminationByDecayModeFindingNewDMs'),
        cut = cms.double(0.5)
    )
)
## ByLooseIsolation
hpsPFTauDiscriminationByLooseIsolation = pfRecoTauDiscriminationByIsolation.clone(
    PFTauProducer = cms.InputTag("hpsPFTauProducer"),
    Prediscriminants = requireDecayMode.clone(),
    ApplyDiscriminationByTrackerIsolation = False,
    ApplyDiscriminationByECALIsolation = True,
    applyOccupancyCut = True
)
hpsPFTauDiscriminationByLooseIsolation.Prediscriminants.preIso = cms.PSet(
    Producer = cms.InputTag("hpsPFTauDiscriminationByLooseChargedIsolation"),
    cut = cms.double(0.5)
)
## ByMediumIsolation
hpsPFTauDiscriminationByMediumIsolation = hpsPFTauDiscriminationByLooseIsolation.clone()
hpsPFTauDiscriminationByMediumIsolation.qualityCuts.isolationQualityCuts.minTrackPt = 0.8
hpsPFTauDiscriminationByMediumIsolation.qualityCuts.isolationQualityCuts.minGammaEt = 1.0
hpsPFTauDiscriminationByMediumIsolation.Prediscriminants.preIso.Producer = cms.InputTag("hpsPFTauDiscriminationByMediumChargedIsolation")
## ByTightIsolation
hpsPFTauDiscriminationByTightIsolation = hpsPFTauDiscriminationByLooseIsolation.clone()
hpsPFTauDiscriminationByTightIsolation.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
hpsPFTauDiscriminationByTightIsolation.qualityCuts.isolationQualityCuts.minGammaEt = 1.0
hpsPFTauDiscriminationByTightIsolation.Prediscriminants.preIso.Producer = cms.InputTag("hpsPFTauDiscriminationByTightChargedIsolation")
## ByLooseIsolationDBSumPtCorr
hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr = hpsPFTauDiscriminationByLooseIsolation.clone(
    deltaBetaPUTrackPtCutOverride     = True, # Set the boolean = True to override.
    deltaBetaPUTrackPtCutOverride_val = 0.5,  # Set the value for new value.
    applyDeltaBetaCorrection = True,
    isoConeSizeForDeltaBeta = 0.8,
    deltaBetaFactor = "%0.4f"%(0.0123/0.1687),
    applyOccupancyCut = False,
    applySumPtCut = True,
)
hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr.maximumSumPtCut = hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt
## ByMediumIsolationDBSumPtCorr
hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr = hpsPFTauDiscriminationByMediumIsolation.clone(
    deltaBetaPUTrackPtCutOverride     = True, # Set the boolean = True to override.
    deltaBetaPUTrackPtCutOverride_val = 0.5,  # Set the value for new value.
    applyDeltaBetaCorrection = True,
    isoConeSizeForDeltaBeta = 0.8,
    deltaBetaFactor = "%0.4f"%(0.0462/0.1687),
    applyOccupancyCut = False,
    applySumPtCut = True,
)
hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr.maximumSumPtCut = hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt
## ByTightIsolationDBSumPtCorr
hpsPFTauDiscriminationByTightIsolationDBSumPtCorr = hpsPFTauDiscriminationByTightIsolation.clone(
    deltaBetaPUTrackPtCutOverride     = True, # Set the boolean = True to override.
    deltaBetaPUTrackPtCutOverride_val = 0.5,  # Set the value for new value.
    applyDeltaBetaCorrection = True,
    isoConeSizeForDeltaBeta = 0.8,
    deltaBetaFactor = "%0.4f"%(ak4dBetaCorrection),
    applyOccupancyCut = False,
    applySumPtCut = True,
)
hpsPFTauDiscriminationByTightIsolationDBSumPtCorr.maximumSumPtCut = hpsPFTauDiscriminationByTightIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt
## ByLooseCombinedIsolationDBSumPtCorr
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr = hpsPFTauDiscriminationByLooseIsolationDBSumPtCorr.clone(
    ApplyDiscriminationByTrackerIsolation = True,
    ApplyDiscriminationByECALIsolation = True,
    deltaBetaFactor = "%0.4f"%(ak4dBetaCorrection),
    applyOccupancyCut = False,
    applySumPtCut = True,
    maximumSumPtCut = 2.5,
    Prediscriminants = requireDecayMode.clone()
)
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
phase2_common.toModify(hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr.qualityCuts,
                       isolationQualityCuts = dict( minTrackPt = 0.8 )
                      )
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt = 1.0
## ByMediumCombinedIsolationDBSumPtCorr
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr = hpsPFTauDiscriminationByMediumIsolationDBSumPtCorr.clone(
    ApplyDiscriminationByTrackerIsolation = True,
    ApplyDiscriminationByECALIsolation = True,
    deltaBetaFactor = "%0.4f"%(ak4dBetaCorrection),
    applyOccupancyCut = False,
    applySumPtCut = True,
    maximumSumPtCut = 1.5,
    Prediscriminants = requireDecayMode.clone()
)
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
phase2_common.toModify(hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr.qualityCuts,
                       isolationQualityCuts = dict( minTrackPt = 0.8 )
                      )
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt = 1.0
## ByTightCombinedIsolationDBSumPtCorr
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr = hpsPFTauDiscriminationByTightIsolationDBSumPtCorr.clone(
    ApplyDiscriminationByTrackerIsolation = True,
    ApplyDiscriminationByECALIsolation = True,
    deltaBetaFactor = "%0.4f"%(ak4dBetaCorrection),
    applyOccupancyCut = False,
    applySumPtCut = True,
    maximumSumPtCut = 0.8,
    Prediscriminants = requireDecayMode.clone()
)
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minTrackPt = 0.5
phase2_common.toModify(hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr.qualityCuts,
                       isolationQualityCuts = dict( minTrackPt = 0.8 )
                      )
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr.qualityCuts.isolationQualityCuts.minGammaEt = 1.0
## ByLooseChargedIsolation
hpsPFTauDiscriminationByLooseChargedIsolation = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr.clone(
    ApplyDiscriminationByECALIsolation = False
)
## ByLooseCombinedIsolationDBSumPtCorr3Hits
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr.clone()
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits.qualityCuts.isolationQualityCuts.minTrackHits = cms.uint32(3)
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits.applyPhotonPtSumOutsideSignalConeCut = cms.bool(True)
## ByMediumCombinedIsolationDBSumPtCorr3Hits
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits = hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr.clone()
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits.qualityCuts.isolationQualityCuts.minTrackHits = cms.uint32(3)
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits.applyPhotonPtSumOutsideSignalConeCut = cms.bool(True)
## ByTightCombinedIsolationDBSumPtCorr3Hits
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits = hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr.clone()
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits.qualityCuts.isolationQualityCuts.minTrackHits = cms.uint32(3)
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits.applyPhotonPtSumOutsideSignalConeCut = cms.bool(True)
## ByRawCombinedIsolationDBSumPtCorr3Hits
hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr3Hits = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits.clone(
    applySumPtCut = False,
    storeRawSumPt = cms.bool(True)
)
## hpsPFTauDiscriminationByCombinedIsolationSeqDBSumPtCorr3Hits 
hpsPFTauDiscriminationByCombinedIsolationDBSumPtCorr3HitsTask = cms.Task(
    hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits,
    hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits,
    hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits,
    hpsPFTauDiscriminationByRawCombinedIsolationDBSumPtCorr3Hits
)
hpsPFTauDiscriminationByCombinedIsolationSeqDBSumPtCorr3Hits = cms.Sequence(
    hpsPFTauDiscriminationByCombinedIsolationDBSumPtCorr3HitsTask
)
## Discrimination ByLooseCombinedIsolationDBSumPtCorr3HitsdR03
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3HitsdR03 = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits.clone()
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3HitsdR03.deltaBetaFactor = cms.string('0.0720') # 0.2*(0.3/0.5)^2
hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3HitsdR03.customOuterCone = cms.double(0.3)
## Discrimination ByMediumCombinedIsolationDBSumPtCorr3HitsdR03
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3HitsdR03 = hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits.clone()
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3HitsdR03.deltaBetaFactor = cms.string('0.0720') # 0.2*(0.3/0.5)^2
hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3HitsdR03.customOuterCone = cms.double(0.3)
## Discrimination ByTightCombinedIsolationDBSumPtCorr3HitsdR03
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3HitsdR03 = hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits.clone()
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3HitsdR03.deltaBetaFactor = cms.string('0.0720') # 0.2*(0.3/0.5)^2
hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3HitsdR03.customOuterCone = cms.double(0.3)
## hpsPFTauDiscriminationByCombinedIsolationSeqDBSumPtCorr3HitsdR03
hpsPFTauDiscriminationByCombinedIsolationDBSumPtCorr3HitsdR03Task = cms.Task(
    hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3HitsdR03,
    hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3HitsdR03,
    hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3HitsdR03
)
hpsPFTauDiscriminationByCombinedIsolationSeqDBSumPtCorr3HitsdR03 = cms.Sequence(
    hpsPFTauDiscriminationByCombinedIsolationDBSumPtCorr3HitsdR03Task
)
## ByLoosePileupWeightedIsolation3Hits (kept for Validation)
hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits.clone(
    ApplyDiscriminationByECALIsolation = cms.bool(False),
    applyDeltaBetaCorrection = cms.bool(False),
    ApplyDiscriminationByWeightedECALIsolation = cms.bool(True),
    UseAllPFCandsForWeights = cms.bool(True),
    applyFootprintCorrection = cms.bool(True),
    applyPhotonPtSumOutsideSignalConeCut = cms.bool(True)    
)
## ByMediumPileupWeightedIsolation3Hits (kept for Validation)
hpsPFTauDiscriminationByMediumPileupWeightedIsolation3Hits = hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits.clone(
    maximumSumPtCut = hpsPFTauDiscriminationByMediumCombinedIsolationDBSumPtCorr3Hits.maximumSumPtCut
)
## ByTightPileupWeightedIsolation3Hits (kept for Validation)
hpsPFTauDiscriminationByTightPileupWeightedIsolation3Hits = hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits.clone(
    maximumSumPtCut = hpsPFTauDiscriminationByTightCombinedIsolationDBSumPtCorr3Hits.maximumSumPtCut
)
## ByRawPileupWeightedIsolation3Hits (kept for Validation)
hpsPFTauDiscriminationByRawPileupWeightedIsolation3Hits = hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits.clone(
    Prediscriminants = cms.PSet(
        BooleanOperator = cms.string("and"),
        decayMode = cms.PSet(
            Producer = cms.InputTag('hpsPFTauDiscriminationByPhotonPtSumOutsideSignalCone'),
            cut = cms.double(0.5)
        )
    ),
    applySumPtCut = cms.bool(False),
    storeRawSumPt = cms.bool(True)
)
## hpsPFTauDiscriminationByPhotonPtSumOutsideSignalCone
hpsPFTauDiscriminationByPhotonPtSumOutsideSignalCone = hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits.clone(
    applySumPtCut = cms.bool(False)
)
## hpsPFTauDiscriminationByPileupWeightedIsolationSeq3Hits
hpsPFTauDiscriminationByPileupWeightedIsolation3HitsTask = cms.Task(
   hpsPFTauDiscriminationByLoosePileupWeightedIsolation3Hits,
   hpsPFTauDiscriminationByMediumPileupWeightedIsolation3Hits,
   hpsPFTauDiscriminationByTightPileupWeightedIsolation3Hits,
   hpsPFTauDiscriminationByPhotonPtSumOutsideSignalCone,
   hpsPFTauDiscriminationByRawPileupWeightedIsolation3Hits
)
hpsPFTauDiscriminationByPileupWeightedIsolationSeq3Hits = cms.Sequence(
    hpsPFTauDiscriminationByPileupWeightedIsolation3HitsTask
)


## ByLooseMuonRejection3
hpsPFTauDiscriminationByLooseMuonRejection3 = pfRecoTauDiscriminationAgainstMuon2.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = noPrediscriminants,
    discriminatorOption = cms.string('custom'),
    maxNumberOfMatches = cms.int32(1),
    doCaloMuonVeto = cms.bool(True),
    maxNumberOfHitsLast2Stations = cms.int32(-1)
)
hpsPFTauDiscriminationByTightMuonRejection3 = hpsPFTauDiscriminationByLooseMuonRejection3.clone(
    maxNumberOfHitsLast2Stations = cms.int32(0)
)


## ByLooseElectronRejection
hpsPFTauDiscriminationByLooseElectronRejection = pfRecoTauDiscriminationAgainstElectron.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = noPrediscriminants,
    PFElectronMVA_maxValue = cms.double(0.6)
)
## ByMediumElectronRejection
hpsPFTauDiscriminationByMediumElectronRejection = pfRecoTauDiscriminationAgainstElectron.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = noPrediscriminants,
    ApplyCut_EcalCrackCut = cms.bool(True)
)
## ByTightElectronRejection
hpsPFTauDiscriminationByTightElectronRejection = pfRecoTauDiscriminationAgainstElectron.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = noPrediscriminants,
    ApplyCut_EcalCrackCut = cms.bool(True),
    ApplyCut_BremCombined = cms.bool(True)
)
## ByDeadECALElectronRejection 
hpsPFTauDiscriminationByDeadECALElectronRejection = pfRecoTauDiscriminationAgainstElectronDeadECAL.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone()
)
## ByMVA6rawElectronRejection
hpsPFTauDiscriminationByMVA6rawElectronRejection = pfRecoTauDiscriminationAgainstElectronMVA6.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
    loadMVAfromDB = cms.bool(True),
    vetoEcalCracks = cms.bool(True),
    mvaName_NoEleMatch_woGwoGSF_BL = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_BL"),
    mvaName_NoEleMatch_wGwoGSF_BL = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_BL"),
    mvaName_woGwGSF_BL = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_BL"),
    mvaName_wGwGSF_BL = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_BL"),
    mvaName_NoEleMatch_woGwoGSF_EC = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_EC"),
    mvaName_NoEleMatch_wGwoGSF_EC = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_EC"),
    mvaName_woGwGSF_EC = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_EC"),
    mvaName_wGwGSF_EC = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_EC")
)
## ByMVA6VLooseElectronRejection
hpsPFTauDiscriminationByMVA6VLooseElectronRejection = recoTauDiscriminantCutMultiplexer.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByMVA6rawElectronRejection'),
    key = cms.InputTag('hpsPFTauDiscriminationByMVA6rawElectronRejection:category'),
    loadMVAfromDB = cms.bool(True),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0), # minMVANoEleMatchWOgWOgsfBL
            cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_BL_WPEff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(2), # minMVANoEleMatchWgWOgsfBL
            cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_BL_WPEff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(5), # minMVAWOgWgsfBL
            cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_BL_WPEff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(7), # minMVAWgWgsfBL
            cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_BL_WPEff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(8), # minMVANoEleMatchWOgWOgsfEC
            cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_EC_WPEff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(10), # minMVANoEleMatchWgWOgsfEC
            cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_EC_WPEff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(13), # minMVAWOgWgsfEC
            cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_EC_WPEff99"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(15), # minMVAWgWgsfEC
            cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_EC_WPEff99"),
            variable = cms.string("pt")
        )
    )
)
## ByMVA6LooseElectronRejection
hpsPFTauDiscriminationByMVA6LooseElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA6VLooseElectronRejection)
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[0].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_BL_WPEff96")
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[1].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_BL_WPEff96")
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[2].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_BL_WPEff96")
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[3].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_BL_WPEff96")
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[4].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_EC_WPEff96")
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[5].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_EC_WPEff96")
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[6].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_EC_WPEff96")
hpsPFTauDiscriminationByMVA6LooseElectronRejection.mapping[7].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_EC_WPEff96")
## ByMVA6VMediumElectronRejection
hpsPFTauDiscriminationByMVA6MediumElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA6VLooseElectronRejection)
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[0].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_BL_WPEff91")
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[1].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_BL_WPEff91")
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[2].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_BL_WPEff91")
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[3].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_BL_WPEff91")
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[4].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_EC_WPEff91")
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[5].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_EC_WPEff91")
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[6].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_EC_WPEff91")
hpsPFTauDiscriminationByMVA6MediumElectronRejection.mapping[7].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_EC_WPEff91")
## ByMVA6TightElectronRejection
hpsPFTauDiscriminationByMVA6TightElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA6VLooseElectronRejection)
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[0].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_BL_WPEff85")
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[1].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_BL_WPEff85")
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[2].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_BL_WPEff85")
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[3].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_BL_WPEff85")
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[4].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_EC_WPEff85")
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[5].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_EC_WPEff85")
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[6].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_EC_WPEff85")
hpsPFTauDiscriminationByMVA6TightElectronRejection.mapping[7].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_EC_WPEff85")
## ByMVA6VTightElectronRejection
hpsPFTauDiscriminationByMVA6VTightElectronRejection = copy.deepcopy(hpsPFTauDiscriminationByMVA6VLooseElectronRejection)
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[0].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_BL_WPEff79")
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[1].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_BL_WPEff79")
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[2].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_BL_WPEff79")
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[3].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_BL_WPEff79")
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[4].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_woGwoGSF_EC_WPEff79")
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[5].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_NoEleMatch_wGwoGSF_EC_WPEff79")
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[6].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_woGwGSF_EC_WPEff79")
hpsPFTauDiscriminationByMVA6VTightElectronRejection.mapping[7].cut = cms.string("RecoTauTag_antiElectronMVA6v1_gbr_wGwGSF_EC_WPEff79")

# Define the HPS selection discriminator used in cleaning
hpsSelectionDiscriminator.PFTauProducer = cms.InputTag("combinatoricRecoTaus")
#----------------------------------------------------------------------------
# CV: disable 3Prong1Pi0 decay mode
hpsSelectionDiscriminator.decayModes = cms.VPSet(
    decayMode_1Prong0Pi0,
    decayMode_1Prong1Pi0,
    decayMode_1Prong2Pi0,
    decayMode_2Prong0Pi0,
    decayMode_2Prong1Pi0,
    decayMode_3Prong0Pi0,
    decayMode_3Prong1Pi0,
)
#----------------------------------------------------------------------------

from RecoTauTag.RecoTau.RecoTauCleaner_cfi import RecoTauCleaner
hpsPFTauProducerSansRefs = RecoTauCleaner.clone(
    src = cms.InputTag("combinatoricRecoTaus")
)
hpsPFTauProducerSansRefs.cleaners[1].src = cms.InputTag("hpsSelectionDiscriminator")

from RecoTauTag.RecoTau.RecoTauPiZeroUnembedder_cfi import RecoTauPiZeroUnembedder
hpsPFTauProducer = RecoTauPiZeroUnembedder.clone(
    src = cms.InputTag("hpsPFTauProducerSansRefs")
)

from RecoTauTag.RecoTau.PFTauPrimaryVertexProducer_cfi      import *
from RecoTauTag.RecoTau.PFTauSecondaryVertexProducer_cfi    import *
from RecoTauTag.RecoTau.PFTauTransverseImpactParameters_cfi import *
hpsPFTauPrimaryVertexProducer = PFTauPrimaryVertexProducer.clone(
    PFTauTag = cms.InputTag("hpsPFTauProducer"),
    ElectronTag = cms.InputTag(""),
    MuonTag = cms.InputTag(""),
    PVTag = cms.InputTag("offlinePrimaryVertices"),
    beamSpot = cms.InputTag("offlineBeamSpot"),
    Algorithm = cms.int32(0),
    useBeamSpot = cms.bool(True),
    RemoveMuonTracks = cms.bool(False),
    RemoveElectronTracks = cms.bool(False),
    useSelectedTaus = cms.bool(False),
    discriminators = cms.VPSet(
        cms.PSet(
            discriminator = cms.InputTag('hpsPFTauDiscriminationByDecayModeFindingNewDMs'),
            selectionCut = cms.double(0.5)
        )
    ),
    cut = cms.string("pt > 18.0 & abs(eta) < 2.4")
)

hpsPFTauSecondaryVertexProducer = PFTauSecondaryVertexProducer.clone(
    PFTauTag = cms.InputTag("hpsPFTauProducer")
)
hpsPFTauTransverseImpactParameters = PFTauTransverseImpactParameters.clone(
    PFTauTag = cms.InputTag("hpsPFTauProducer"),
    PFTauPVATag = cms.InputTag("hpsPFTauPrimaryVertexProducer"),
    PFTauSVATag = cms.InputTag("hpsPFTauSecondaryVertexProducer"),
    useFullCalculation = cms.bool(True)
)
hpsPFTauVertexAndImpactParametersTask = cms.Task(
    hpsPFTauPrimaryVertexProducer,
    hpsPFTauSecondaryVertexProducer,
    hpsPFTauTransverseImpactParameters
)
hpsPFTauVertexAndImpactParametersSeq = cms.Sequence(
    hpsPFTauVertexAndImpactParametersTask
)
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByMVAIsolation2_cff import *
hpsPFTauChargedIsoPtSum = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
    ApplyDiscriminationByECALIsolation = cms.bool(False),
    ApplyDiscriminationByTrackerIsolation = cms.bool(True),
    applySumPtCut = cms.bool(False),
    applyDeltaBetaCorrection = cms.bool(False),
    storeRawSumPt = cms.bool(True),
    storeRawPUsumPt = cms.bool(False),     
    customOuterCone = PFRecoTauPFJetInputs.isolationConeSize,
    isoConeSizeForDeltaBeta = cms.double(0.8),
    verbosity = cms.int32(0)
)
hpsPFTauNeutralIsoPtSum = hpsPFTauChargedIsoPtSum.clone(
    ApplyDiscriminationByECALIsolation = cms.bool(True),
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    verbosity = cms.int32(0)
)
hpsPFTauPUcorrPtSum = hpsPFTauChargedIsoPtSum.clone(
    ApplyDiscriminationByECALIsolation = cms.bool(False),
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    applyDeltaBetaCorrection = cms.bool(True),
    storeRawSumPt = cms.bool(False),
    storeRawPUsumPt = cms.bool(True),
    verbosity = cms.int32(0)
)
hpsPFTauNeutralIsoPtSumWeight = hpsPFTauChargedIsoPtSum.clone(
    ApplyDiscriminationByWeightedECALIsolation = cms.bool(True),
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    UseAllPFCandsForWeights = cms.bool(True),
    verbosity = cms.int32(0)
)
hpsPFTauFootprintCorrection = hpsPFTauChargedIsoPtSum.clone(    
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    storeRawSumPt = cms.bool(False),
    storeRawFootprintCorrection = cms.bool(True),
    verbosity = cms.int32(0)
)
hpsPFTauPhotonPtSumOutsideSignalCone = hpsPFTauChargedIsoPtSum.clone(    
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    storeRawSumPt = cms.bool(False),
    storeRawPhotonSumPt_outsideSignalCone = cms.bool(True),
    verbosity = cms.int32(0)
)

#Define new Run2 MVA isolations
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByMVAIsolationRun2_cff import *
hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw = discriminationByIsolationMVArun2v1raw.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
    loadMVAfromDB = cms.bool(True),
    mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2"),
    mvaOpt = cms.string("DBoldDMwLTwGJ"),
    srcTauTransverseImpactParameters = cms.InputTag('hpsPFTauTransverseImpactParameters'),
    srcChargedIsoPtSum = cms.InputTag('hpsPFTauChargedIsoPtSum'),
    srcNeutralIsoPtSum = cms.InputTag('hpsPFTauNeutralIsoPtSum'),
    srcPUcorrPtSum = cms.InputTag('hpsPFTauPUcorrPtSum'),
    srcPhotonPtSumOutsideSignalCone = cms.InputTag('hpsPFTauPhotonPtSumOutsideSignalCone'),
    srcFootprintCorrection = cms.InputTag('hpsPFTauFootprintCorrection'),
    verbosity = cms.int32(0)
)

hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT = discriminationByIsolationMVArun2v1VLoose.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw:category'),
    loadMVAfromDB = cms.bool(True),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT.clone()
hpsPFTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff95")
hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT.clone()
hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff80")
hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT.clone()
hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff70")
hpsPFTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT.clone()
hpsPFTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff60")
hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT.clone()
hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff50")
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT.clone()
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT2017v2_WPEff40")

hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw = hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw.clone(
    mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2"),
    mvaOpt = cms.string("DBnewDMwLTwGJ"),
    verbosity = cms.int32(0)
)

hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT.clone(
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw:category'),
    loadMVAfromDB = cms.bool(True),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBnewDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT.clone()
hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBnewDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff80")
hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBnewDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT.clone()
hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBnewDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff70")
hpsPFTauDiscriminationByTightIsolationMVArun2v1DBnewDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT.clone()
hpsPFTauDiscriminationByTightIsolationMVArun2v1DBnewDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff60")
hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBnewDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT.clone()
hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBnewDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff50")
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBnewDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT.clone()
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBnewDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT2017v2_WPEff40")

hpsPFTauDiscriminationByIsolationMVArun2v1PWoldDMwLTraw = hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw.clone(
    mvaName = cms.string("RecoTauTag_tauIdMVAPWoldDMwLTv1"),
    mvaOpt = cms.string("PWoldDMwLT"),
    srcNeutralIsoPtSum = cms.InputTag('hpsPFTauNeutralIsoPtSumWeight'),
    verbosity = cms.int32(0)
)

hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWoldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT.clone(
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1PWoldDMwLTraw'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1PWoldDMwLTraw:category'),
    loadMVAfromDB = cms.bool(True),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAPWoldDMwLTv1_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVAPWoldDMwLTv1_WPEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByLooseIsolationMVArun2v1PWoldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWoldDMwLT.clone()
hpsPFTauDiscriminationByLooseIsolationMVArun2v1PWoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWoldDMwLTv1_WPEff80")
hpsPFTauDiscriminationByMediumIsolationMVArun2v1PWoldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWoldDMwLT.clone()
hpsPFTauDiscriminationByMediumIsolationMVArun2v1PWoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWoldDMwLTv1_WPEff70")
hpsPFTauDiscriminationByTightIsolationMVArun2v1PWoldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWoldDMwLT.clone()
hpsPFTauDiscriminationByTightIsolationMVArun2v1PWoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWoldDMwLTv1_WPEff60")
hpsPFTauDiscriminationByVTightIsolationMVArun2v1PWoldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWoldDMwLT.clone()
hpsPFTauDiscriminationByVTightIsolationMVArun2v1PWoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWoldDMwLTv1_WPEff50")
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1PWoldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWoldDMwLT.clone()
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1PWoldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWoldDMwLTv1_WPEff40")

hpsPFTauDiscriminationByIsolationMVArun2v1PWnewDMwLTraw = hpsPFTauDiscriminationByIsolationMVArun2v1PWoldDMwLTraw.clone(
    mvaName = cms.string("RecoTauTag_tauIdMVAPWnewDMwLTv1"),
    mvaOpt = cms.string("PWnewDMwLT"),
    verbosity = cms.int32(0)
)

hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWnewDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWoldDMwLT.clone(
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1PWnewDMwLTraw'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1PWnewDMwLTraw:category'),
    loadMVAfromDB = cms.bool(True),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAPWnewDMwLTv1_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVAPWnewDMwLTv1_WPEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByLooseIsolationMVArun2v1PWnewDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWnewDMwLT.clone()
hpsPFTauDiscriminationByLooseIsolationMVArun2v1PWnewDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWnewDMwLTv1_WPEff80")
hpsPFTauDiscriminationByMediumIsolationMVArun2v1PWnewDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWnewDMwLT.clone()
hpsPFTauDiscriminationByMediumIsolationMVArun2v1PWnewDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWnewDMwLTv1_WPEff70")
hpsPFTauDiscriminationByTightIsolationMVArun2v1PWnewDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWnewDMwLT.clone()
hpsPFTauDiscriminationByTightIsolationMVArun2v1PWnewDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWnewDMwLTv1_WPEff60")
hpsPFTauDiscriminationByVTightIsolationMVArun2v1PWnewDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWnewDMwLT.clone()
hpsPFTauDiscriminationByVTightIsolationMVArun2v1PWnewDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWnewDMwLTv1_WPEff50")
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1PWnewDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWnewDMwLT.clone()
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1PWnewDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWnewDMwLTv1_WPEff40")

hpsPFTauChargedIsoPtSumdR03 = hpsPFTauDiscriminationByLooseCombinedIsolationDBSumPtCorr3Hits.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
    ApplyDiscriminationByECALIsolation = cms.bool(False),
    ApplyDiscriminationByTrackerIsolation = cms.bool(True),
    applySumPtCut = cms.bool(False),
    applyDeltaBetaCorrection = cms.bool(False),
    storeRawSumPt = cms.bool(True),
    storeRawPUsumPt = cms.bool(False),
    customOuterCone = cms.double(0.3),
    isoConeSizeForDeltaBeta = cms.double(0.8),
    verbosity = cms.int32(0)
)
hpsPFTauNeutralIsoPtSumdR03 = hpsPFTauChargedIsoPtSumdR03.clone(
    ApplyDiscriminationByECALIsolation = cms.bool(True),
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    verbosity = cms.int32(0)
)
hpsPFTauPUcorrPtSumdR03 = hpsPFTauChargedIsoPtSumdR03.clone(
    ApplyDiscriminationByECALIsolation = cms.bool(False),
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    applyDeltaBetaCorrection = cms.bool(True),
    storeRawSumPt = cms.bool(False),
    storeRawPUsumPt = cms.bool(True),
    verbosity = cms.int32(0)
)
hpsPFTauNeutralIsoPtSumWeightdR03 = hpsPFTauChargedIsoPtSumdR03.clone(
    ApplyDiscriminationByWeightedECALIsolation = cms.bool(True),
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    UseAllPFCandsForWeights = cms.bool(True),
    verbosity = cms.int32(0)
)
hpsPFTauFootprintCorrectiondR03 = hpsPFTauChargedIsoPtSumdR03.clone(
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    storeRawSumPt = cms.bool(False),
    storeRawFootprintCorrection = cms.bool(True),
    verbosity = cms.int32(0)
)
hpsPFTauPhotonPtSumOutsideSignalConedR03 = hpsPFTauChargedIsoPtSumdR03.clone(
    ApplyDiscriminationByTrackerIsolation = cms.bool(False),
    storeRawSumPt = cms.bool(False),
    storeRawPhotonSumPt_outsideSignalCone = cms.bool(True),
    verbosity = cms.int32(0)
)

hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLTraw = hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw.clone(
    mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2"),
    mvaOpt = cms.string("DBoldDMwLTwGJ"),
    srcChargedIsoPtSum = cms.InputTag('hpsPFTauChargedIsoPtSumdR03'),
    srcNeutralIsoPtSum = cms.InputTag('hpsPFTauNeutralIsoPtSumdR03'),
    srcPUcorrPtSum = cms.InputTag('hpsPFTauPUcorrPtSumdR03'),
    srcPhotonPtSumOutsideSignalCone = cms.InputTag('hpsPFTauPhotonPtSumOutsideSignalConedR03'),
    srcFootprintCorrection = cms.InputTag('hpsPFTauFootprintCorrectiondR03'),
    verbosity = cms.int32(0)
)
hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBdR03oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLTraw'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLTraw:category'),
    loadMVAfromDB = cms.bool(True),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBdR03oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBdR03oldDMwLT.clone()
hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBdR03oldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff80")
hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBdR03oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBdR03oldDMwLT.clone()
hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBdR03oldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff70")
hpsPFTauDiscriminationByTightIsolationMVArun2v1DBdR03oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBdR03oldDMwLT.clone()
hpsPFTauDiscriminationByTightIsolationMVArun2v1DBdR03oldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff60")
hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBdR03oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBdR03oldDMwLT.clone()
hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBdR03oldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff50")
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBdR03oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBdR03oldDMwLT.clone()
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBdR03oldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT2017v2_WPEff40")

hpsPFTauDiscriminationByIsolationMVArun2v1PWdR03oldDMwLTraw = hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLTraw.clone(
    mvaName = cms.string("RecoTauTag_tauIdMVAPWdR03oldDMwLTv1"),
    mvaOpt = cms.string("PWoldDMwLT"),
    srcNeutralIsoPtSum = cms.InputTag('hpsPFTauNeutralIsoPtSumWeightdR03'),
    verbosity = cms.int32(0)
)
hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWdR03oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBdR03oldDMwLT.clone(
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1PWdR03oldDMwLTraw'),
    key = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1PWdR03oldDMwLTraw:category'),
    loadMVAfromDB = cms.bool(True),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAPWdR03oldDMwLTv1_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVAPWdR03oldDMwLTv1_WPEff90"),
            variable = cms.string("pt")
        )
    )
)
hpsPFTauDiscriminationByLooseIsolationMVArun2v1PWdR03oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWdR03oldDMwLT.clone()
hpsPFTauDiscriminationByLooseIsolationMVArun2v1PWdR03oldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWdR03oldDMwLTv1_WPEff80")
hpsPFTauDiscriminationByMediumIsolationMVArun2v1PWdR03oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWdR03oldDMwLT.clone()
hpsPFTauDiscriminationByMediumIsolationMVArun2v1PWdR03oldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWdR03oldDMwLTv1_WPEff70")
hpsPFTauDiscriminationByTightIsolationMVArun2v1PWdR03oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWdR03oldDMwLT.clone()
hpsPFTauDiscriminationByTightIsolationMVArun2v1PWdR03oldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWdR03oldDMwLTv1_WPEff60")
hpsPFTauDiscriminationByVTightIsolationMVArun2v1PWdR03oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWdR03oldDMwLT.clone()
hpsPFTauDiscriminationByVTightIsolationMVArun2v1PWdR03oldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWdR03oldDMwLTv1_WPEff50")
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1PWdR03oldDMwLT = hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWdR03oldDMwLT.clone()
hpsPFTauDiscriminationByVVTightIsolationMVArun2v1PWdR03oldDMwLT.mapping[0].cut = cms.string("RecoTauTag_tauIdMVAPWdR03oldDMwLTv1_WPEff40")

hpsPFTauIsolationSumsTask = cms.Task(
    hpsPFTauChargedIsoPtSum,
    hpsPFTauNeutralIsoPtSum,
    hpsPFTauPUcorrPtSum,
    hpsPFTauNeutralIsoPtSumWeight,
    hpsPFTauFootprintCorrection,
    hpsPFTauPhotonPtSumOutsideSignalCone,    
    )

hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTTask = cms.Task(
    hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw,
    hpsPFTauDiscriminationByVVLooseIsolationMVArun2v1DBoldDMwLT,
    hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBoldDMwLT,
    hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBoldDMwLT,
    hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBoldDMwLT,
    hpsPFTauDiscriminationByTightIsolationMVArun2v1DBoldDMwLT,
    hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBoldDMwLT,
    hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBoldDMwLT
    )

hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTTask = cms.Task(
    hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw,
    hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBnewDMwLT,
    hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBnewDMwLT,
    hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBnewDMwLT,
    hpsPFTauDiscriminationByTightIsolationMVArun2v1DBnewDMwLT,
    hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBnewDMwLT,
    hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBnewDMwLT
    )

hpsPFTauDiscriminationByIsolationMVArun2v1PWoldDMwLTTask = cms.Task(
    hpsPFTauDiscriminationByIsolationMVArun2v1PWoldDMwLTraw,
    hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWoldDMwLT,
    hpsPFTauDiscriminationByLooseIsolationMVArun2v1PWoldDMwLT,
    hpsPFTauDiscriminationByMediumIsolationMVArun2v1PWoldDMwLT,
    hpsPFTauDiscriminationByTightIsolationMVArun2v1PWoldDMwLT,
    hpsPFTauDiscriminationByVTightIsolationMVArun2v1PWoldDMwLT,
    hpsPFTauDiscriminationByVVTightIsolationMVArun2v1PWoldDMwLT
    )

hpsPFTauDiscriminationByIsolationMVArun2v1PWnewDMwLTTask = cms.Task(
    hpsPFTauDiscriminationByIsolationMVArun2v1PWnewDMwLTraw,
    hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWnewDMwLT,
    hpsPFTauDiscriminationByLooseIsolationMVArun2v1PWnewDMwLT,
    hpsPFTauDiscriminationByMediumIsolationMVArun2v1PWnewDMwLT,
    hpsPFTauDiscriminationByTightIsolationMVArun2v1PWnewDMwLT,
    hpsPFTauDiscriminationByVTightIsolationMVArun2v1PWnewDMwLT,
    hpsPFTauDiscriminationByVVTightIsolationMVArun2v1PWnewDMwLT,
    )

hpsPFTauIsolationSums03Task = cms.Task(
    hpsPFTauChargedIsoPtSumdR03,
    hpsPFTauNeutralIsoPtSumdR03,
    hpsPFTauPUcorrPtSumdR03,
    hpsPFTauNeutralIsoPtSumWeightdR03,
    hpsPFTauFootprintCorrectiondR03,
    hpsPFTauPhotonPtSumOutsideSignalConedR03
    )

hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLTTask = cms.Task(
    hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLTraw,
    hpsPFTauDiscriminationByVLooseIsolationMVArun2v1DBdR03oldDMwLT,
    hpsPFTauDiscriminationByLooseIsolationMVArun2v1DBdR03oldDMwLT,
    hpsPFTauDiscriminationByMediumIsolationMVArun2v1DBdR03oldDMwLT,
    hpsPFTauDiscriminationByTightIsolationMVArun2v1DBdR03oldDMwLT,
    hpsPFTauDiscriminationByVTightIsolationMVArun2v1DBdR03oldDMwLT,
    hpsPFTauDiscriminationByVVTightIsolationMVArun2v1DBdR03oldDMwLT
    )

hpsPFTauDiscriminationByIsolationMVArun2v1PWdR03oldDMwLTTask = cms.Task(
    hpsPFTauDiscriminationByIsolationMVArun2v1PWdR03oldDMwLTraw,
    hpsPFTauDiscriminationByVLooseIsolationMVArun2v1PWdR03oldDMwLT,
    hpsPFTauDiscriminationByLooseIsolationMVArun2v1PWdR03oldDMwLT,
    hpsPFTauDiscriminationByMediumIsolationMVArun2v1PWdR03oldDMwLT,
    hpsPFTauDiscriminationByTightIsolationMVArun2v1PWdR03oldDMwLT,
    hpsPFTauDiscriminationByVTightIsolationMVArun2v1PWdR03oldDMwLT,
    hpsPFTauDiscriminationByVVTightIsolationMVArun2v1PWdR03oldDMwLT
    )

hpsPFTauMVAIsolation2Task = cms.Task(
    hpsPFTauIsolationSumsTask,
    hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTTask,
    hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTTask,
    hpsPFTauDiscriminationByIsolationMVArun2v1PWoldDMwLTTask,    
    hpsPFTauDiscriminationByIsolationMVArun2v1PWnewDMwLTTask,
    hpsPFTauIsolationSums03Task,
    hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLTTask,
    hpsPFTauDiscriminationByIsolationMVArun2v1PWdR03oldDMwLTTask
    )

hpsPFTauMVAIsolation2Seq = cms.Sequence(
    hpsPFTauMVAIsolation2Task
    )

produceHPSPFTausTask = cms.Task(
    hpsSelectionDiscriminator,
    #hpsTightIsolationCleaner,
    #hpsMediumIsolationCleaner,
    #hpsLooseIsolationCleaner,
    #hpsVLooseIsolationCleaner,
    hpsPFTauProducerSansRefs,
    hpsPFTauProducer
    )

produceHPSPFTaus = cms.Sequence(
    produceHPSPFTausTask
    )

produceAndDiscriminateHPSPFTausTask = cms.Task(
    produceHPSPFTausTask,
    hpsPFTauDiscriminationByDecayModeFindingNewDMs,
    hpsPFTauDiscriminationByDecayModeFindingOldDMs,
    hpsPFTauDiscriminationByDecayModeFinding, # CV: kept for backwards compatibility
    hpsPFTauDiscriminationByLooseChargedIsolation,
    hpsPFTauDiscriminationByLooseIsolation,
    hpsPFTauDiscriminationByCombinedIsolationDBSumPtCorr3HitsTask,
    hpsPFTauDiscriminationByCombinedIsolationDBSumPtCorr3HitsdR03Task,
    hpsPFTauDiscriminationByPileupWeightedIsolation3HitsTask,
    hpsPFTauDiscriminationByLooseElectronRejection,
    hpsPFTauDiscriminationByMediumElectronRejection,
    hpsPFTauDiscriminationByTightElectronRejection,
    hpsPFTauDiscriminationByMVA6rawElectronRejection,
    hpsPFTauDiscriminationByMVA6VLooseElectronRejection,
    hpsPFTauDiscriminationByMVA6LooseElectronRejection,
    hpsPFTauDiscriminationByMVA6MediumElectronRejection,
    hpsPFTauDiscriminationByMVA6TightElectronRejection,
    hpsPFTauDiscriminationByMVA6VTightElectronRejection,
    hpsPFTauDiscriminationByDeadECALElectronRejection,
    hpsPFTauDiscriminationByLooseMuonRejection3,
    hpsPFTauDiscriminationByTightMuonRejection3,
    hpsPFTauVertexAndImpactParametersTask,
    hpsPFTauMVAIsolation2Task
    )

produceAndDiscriminateHPSPFTaus = cms.Sequence(
    produceAndDiscriminateHPSPFTausTask
    )
