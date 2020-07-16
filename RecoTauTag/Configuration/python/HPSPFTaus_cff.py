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
from RecoTauTag.RecoTau.pfRecoTauDiscriminationAgainstElectronDeadECAL_cfi          import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuon_cfi                      import *
from RecoTauTag.RecoTau.pfRecoTauDiscriminationAgainstMuon2Container_cfi            import *
from RecoTauTag.RecoTau.PFRecoTauDiscriminationAgainstMuonMVA_cfi                   import *

from RecoTauTag.RecoTau.recoTauDiscriminantCutMultiplexerDefault_cfi import recoTauDiscriminantCutMultiplexerDefault
## Helper functions to change the source of the discriminants
from RecoTauTag.RecoTau.TauDiscriminatorTools import *
## PFjet input parameters
from RecoTauTag.RecoTau.PFRecoTauPFJetInputs_cfi import PFRecoTauPFJetInputs
## DeltaBeta correction factor
ak4dBetaCorrection = 0.20

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

## Cut based isolations dR=0.5
hpsPFTauBasicDiscriminators = pfRecoTauDiscriminationByIsolation.clone(
    PFTauProducer = cms.InputTag("hpsPFTauProducer"),
    Prediscriminants = requireDecayMode.clone(),
    deltaBetaPUTrackPtCutOverride     = True, # Set the boolean = True to override.
    deltaBetaPUTrackPtCutOverride_val = 0.5,  # Set the value for new value.
    customOuterCone = PFRecoTauPFJetInputs.isolationConeSize,
    isoConeSizeForDeltaBeta = 0.8,
    deltaBetaFactor = "%0.4f"%(ak4dBetaCorrection),
    qualityCuts = dict(isolationQualityCuts = dict(minTrackHits = 3, minGammaEt = 1.0, minTrackPt = 0.5)),
    IDdefinitions = cms.VPSet(
        cms.PSet(
            IDname = cms.string("ChargedIsoPtSum"),
            ApplyDiscriminationByTrackerIsolation = cms.bool(True),
            storeRawSumPt = cms.bool(True)
            ),
        cms.PSet(
            IDname = cms.string("NeutralIsoPtSum"),
            ApplyDiscriminationByECALIsolation = cms.bool(True),
            storeRawSumPt = cms.bool(True)
            ),
        cms.PSet(
            IDname = cms.string("NeutralIsoPtSumWeight"),
            ApplyDiscriminationByWeightedECALIsolation = cms.bool(True),
            storeRawSumPt = cms.bool(True),
            UseAllPFCandsForWeights = cms.bool(True)
            ),
        cms.PSet(
            IDname = cms.string("TauFootprintCorrection"),
            storeRawFootprintCorrection = cms.bool(True)
            ),
        cms.PSet(
            IDname = cms.string("PhotonPtSumOutsideSignalCone"),
            storeRawPhotonSumPt_outsideSignalCone = cms.bool(True)
            ),
        cms.PSet(
            IDname = cms.string("PUcorrPtSum"),
            applyDeltaBetaCorrection = cms.bool(True),
            storeRawPUsumPt = cms.bool(True)
            ),
        cms.PSet(
            IDname = cms.string("ByRawCombinedIsolationDBSumPtCorr3Hits"),
            ApplyDiscriminationByTrackerIsolation = cms.bool(True),
            ApplyDiscriminationByECALIsolation = cms.bool(True),
            applyDeltaBetaCorrection = cms.bool(True),
            storeRawSumPt = cms.bool(True)
            )
        ),
    IDWPdefinitions = cms.VPSet(
        cms.PSet(
            IDname = cms.string("ByLooseCombinedIsolationDBSumPtCorr3Hits"),
            referenceRawIDNames = cms.vstring("ByRawCombinedIsolationDBSumPtCorr3Hits", "PhotonPtSumOutsideSignalCone"),
            maximumAbsoluteValues = cms.vdouble(2.5, 1.e+9),
            maximumRelativeValues = cms.vdouble(-1.0, 0.10)
            ),
        cms.PSet(
            IDname = cms.string("ByMediumCombinedIsolationDBSumPtCorr3Hits"),
            referenceRawIDNames = cms.vstring("ByRawCombinedIsolationDBSumPtCorr3Hits", "PhotonPtSumOutsideSignalCone"),
            maximumAbsoluteValues = cms.vdouble(1.5, 1.e+9),
            maximumRelativeValues = cms.vdouble(-1.0, 0.10)
            ),
        cms.PSet(
            IDname = cms.string("ByTightCombinedIsolationDBSumPtCorr3Hits"),
            referenceRawIDNames = cms.vstring("ByRawCombinedIsolationDBSumPtCorr3Hits", "PhotonPtSumOutsideSignalCone"),
            maximumAbsoluteValues = cms.vdouble(0.8, 1.e+9),
            maximumRelativeValues = cms.vdouble(-1.0, 0.10)
            ),
        cms.PSet(
            IDname = cms.string("ByLooseChargedIsolation"),
            referenceRawIDNames = cms.vstring("ChargedIsoPtSum"),
            maximumAbsoluteValues = cms.vdouble(2.5)
            ),
        cms.PSet(
            IDname = cms.string("ByPhotonPtSumOutsideSignalCone"),
            referenceRawIDNames = cms.vstring("PhotonPtSumOutsideSignalCone"),
            maximumRelativeValues = cms.vdouble(0.10)
            )
        )
)
phase2_common.toModify(hpsPFTauBasicDiscriminators.qualityCuts,
                       isolationQualityCuts = dict( minTrackPt = 0.8 )
)
hpsPFTauBasicDiscriminatorsTask = cms.Task(
    hpsPFTauBasicDiscriminators
)

## Cut based isolations dR=0.3
hpsPFTauBasicDiscriminatorsdR03 = hpsPFTauBasicDiscriminators.clone(
    deltaBetaFactor = '0.0720', # 0.2*(0.3/0.5)^2
    customOuterCone = 0.3
)
del hpsPFTauBasicDiscriminatorsdR03.IDWPdefinitions[-1] # ByPhotonPtSumOutsideSignalCone not defined for dR03
del hpsPFTauBasicDiscriminatorsdR03.IDWPdefinitions[-1] # ByLooseChargedIsolation not defined for dR03
for pset in hpsPFTauBasicDiscriminatorsdR03.IDdefinitions:
    pset.IDname = pset.IDname.value() + "dR03"
for pset in hpsPFTauBasicDiscriminatorsdR03.IDWPdefinitions:
    pset.IDname = pset.IDname.value() + "dR03"
    pset.referenceRawIDNames = [name + "dR03" for name in pset.referenceRawIDNames.value()]
hpsPFTauBasicDiscriminatorsdR03Task = cms.Task(
    hpsPFTauBasicDiscriminatorsdR03
)

# define helper function to read indices of basic IDs or antimuon
## MuonRejection3
hpsPFTauDiscriminationByMuonRejection3 = pfRecoTauDiscriminationAgainstMuon2Container.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = noPrediscriminants,
    IDWPdefinitions = cms.VPSet(
        cms.PSet(
            IDname = cms.string('ByLooseMuonRejection3'),
            discriminatorOption = cms.string('custom'),
            HoPMin = cms.double(0.2),
            maxNumberOfMatches = cms.int32(1),
            doCaloMuonVeto = cms.bool(True),
            maxNumberOfHitsLast2Stations = cms.int32(-1)
        ),
        cms.PSet(
            IDname = cms.string('ByTightMuonRejection3'),
            discriminatorOption = cms.string('custom'),
            HoPMin = cms.double(0.2),
            maxNumberOfMatches = cms.int32(1),
            doCaloMuonVeto = cms.bool(True),
            maxNumberOfHitsLast2Stations = cms.int32(0)
        )
    )
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
    vetoEcalCracks = cms.bool(False),
    mvaName_NoEleMatch_woGwoGSF_BL = cms.string("RecoTauTag_antiElectronMVA_NoEleMatch_woGwoGSF_BL"),
    mvaName_NoEleMatch_wGwoGSF_BL = cms.string("RecoTauTag_antiElectronMVA_NoEleMatch_wGwoGSF_BL"),
    mvaName_woGwGSF_BL = cms.string("RecoTauTag_antiElectronMVA_woGwGSF_BL"),
    mvaName_wGwGSF_BL = cms.string("RecoTauTag_antiElectronMVA_wGwGSF_BL"),
    mvaName_NoEleMatch_woGwoGSF_EC = cms.string("RecoTauTag_antiElectronMVA_NoEleMatch_woGwoGSF_EC"),
    mvaName_NoEleMatch_wGwoGSF_EC = cms.string("RecoTauTag_antiElectronMVA_NoEleMatch_wGwoGSF_EC"),
    mvaName_woGwGSF_EC = cms.string("RecoTauTag_antiElectronMVA_woGwGSF_EC"),
    mvaName_wGwGSF_EC = cms.string("RecoTauTag_antiElectronMVA_wGwGSF_EC")
)
## ByMVA6ElectronRejection
hpsPFTauDiscriminationByMVA6ElectronRejection = recoTauDiscriminantCutMultiplexerDefault.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByMVA6rawElectronRejection'),
    loadMVAfromDB = cms.bool(True),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0), # minMVANoEleMatchWOgWOgsfBL
            cut = cms.string("RecoTauTag_antiElectronMVA_NoEleMatch_woGwoGSF_BL"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(2), # minMVANoEleMatchWgWOgsfBL
            cut = cms.string("RecoTauTag_antiElectronMVA_NoEleMatch_wGwoGSF_BL"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(5), # minMVAWOgWgsfBL
            cut = cms.string("RecoTauTag_antiElectronMVA_woGwGSF_BL"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(7), # minMVAWgWgsfBL
            cut = cms.string("RecoTauTag_antiElectronMVA_wGwGSF_BL"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(8), # minMVANoEleMatchWOgWOgsfEC
            cut = cms.string("RecoTauTag_antiElectronMVA_NoEleMatch_woGwoGSF_EC"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(10), # minMVANoEleMatchWgWOgsfEC
            cut = cms.string("RecoTauTag_antiElectronMVA_NoEleMatch_wGwoGSF_EC"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(13), # minMVAWOgWgsfEC
            cut = cms.string("RecoTauTag_antiElectronMVA_woGwGSF_EC"),
            variable = cms.string("pt")
        ),
        cms.PSet(
            category = cms.uint32(15), # minMVAWgWgsfEC
            cut = cms.string("RecoTauTag_antiElectronMVA_wGwGSF_EC"),
            variable = cms.string("pt")
        )
    ),
    rawValues = cms.vstring(
        "discriminator",
        "category"
    ),
    workingPoints = cms.vstring(
        "_VLoose",
        "_Loose",
        "_Medium",
        "_Tight",
        "_VTight"
    )
)

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

#Define new Run2 MVA isolations
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByMVAIsolationRun2_cff import *
hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw = discriminationByIsolationMVArun2v1raw.clone(
    PFTauProducer = "hpsPFTauProducer",
    Prediscriminants = requireDecayMode.clone(),
    mvaName = "RecoTauTag_tauIdMVAIsoDBoldDMwLT",
    mvaOpt = "DBoldDMwLTwGJ",
    srcTauTransverseImpactParameters = "hpsPFTauTransverseImpactParameters",
    verbosity = 0
)

hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLT = discriminationByIsolationMVArun2v1.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw'),
    loadMVAfromDB = cms.bool(True),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMwLT"),
            variable = cms.string("pt")
        )
    ),
    workingPoints = cms.vstring(
        "_VVLoose",
        "_VLoose",
        "_Loose",
        "_Medium",
        "_Tight",
        "_VTight",
        "_VVTight"
    )
)
    
hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw = hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw.clone(
    mvaName = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT"),
    mvaOpt = cms.string("DBnewDMwLTwGJ"),
    verbosity = cms.int32(0)
)

hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLT = hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLT.clone(
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw'),
    loadMVAfromDB = cms.bool(True),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVAIsoDBnewDMwLT"),
            variable = cms.string("pt")
        )
    )
)

hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLTraw = hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw.clone(
    mvaName = "RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT",
    mvaOpt = "DBoldDMwLTwGJ",
    srcBasicTauDiscriminators = "hpsPFTauBasicDiscriminatorsdR03",
    inputIDNameSuffix = cms.string("dR03"),
    verbosity = 0
)
hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLT = hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLT.clone(
    PFTauProducer = cms.InputTag('hpsPFTauProducer'),
    Prediscriminants = requireDecayMode.clone(),
    toMultiplex = cms.InputTag('hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLTraw'),
    loadMVAfromDB = cms.bool(True),
    mvaOutput_normalization = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT_mvaOutput_normalization"),
    mapping = cms.VPSet(
        cms.PSet(
            category = cms.uint32(0),
            cut = cms.string("RecoTauTag_tauIdMVAIsoDBoldDMdR0p3wLT"),
            variable = cms.string("pt")
        )
    )
)

hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTTask = cms.Task(
    hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTraw,
    hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLT
    )

hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTTask = cms.Task(
    hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTraw,
    hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLT
    )

hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLTTask = cms.Task(
    hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLTraw,
    hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLT
    )

hpsPFTauMVAIsolation2Task = cms.Task(
    #hpsPFTauBasicDiscriminatorsTask, included separately in produceAndDiscriminateHPSPFTausTask
    hpsPFTauDiscriminationByIsolationMVArun2v1DBoldDMwLTTask,
    hpsPFTauDiscriminationByIsolationMVArun2v1DBnewDMwLTTask,
    #hpsPFTauBasicDiscriminatorsdR03Task, included separately in produceAndDiscriminateHPSPFTausTask
    hpsPFTauDiscriminationByIsolationMVArun2v1DBdR03oldDMwLTTask,
    )

hpsPFTauMVAIsolation2Seq = cms.Sequence(
    hpsPFTauBasicDiscriminatorsTask,
    hpsPFTauBasicDiscriminatorsdR03Task,
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
    hpsPFTauBasicDiscriminatorsTask,
    hpsPFTauBasicDiscriminatorsdR03Task,
    hpsPFTauDiscriminationByLooseElectronRejection,
    hpsPFTauDiscriminationByMediumElectronRejection,
    hpsPFTauDiscriminationByTightElectronRejection,
    hpsPFTauDiscriminationByMVA6rawElectronRejection,
    hpsPFTauDiscriminationByMVA6ElectronRejection,
    hpsPFTauDiscriminationByDeadECALElectronRejection,
    hpsPFTauDiscriminationByMuonRejection3,
    hpsPFTauVertexAndImpactParametersTask,
    hpsPFTauMVAIsolation2Task
    )

produceAndDiscriminateHPSPFTaus = cms.Sequence(
    produceAndDiscriminateHPSPFTausTask
    )
