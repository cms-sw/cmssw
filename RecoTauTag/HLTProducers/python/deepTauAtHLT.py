import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi import *
from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByHPSSelection_cfi import hpsSelectionDiscriminator

from RecoTauTag.RecoTau.PFTauPrimaryVertexProducer_cfi      import *
from RecoTauTag.RecoTau.PFTauSecondaryVertexProducer_cfi    import *
from RecoTauTag.RecoTau.PFTauTransverseImpactParameters_cfi import *

from RecoTauTag.RecoTau.DeepTau_cfi import *

from RecoTauTag.RecoTau.PFRecoTauPFJetInputs_cfi import PFRecoTauPFJetInputs
## DeltaBeta correction factor
ak4dBetaCorrection = 0.20

def update(process):
    process.options.wantSummary = cms.untracked.bool(True)

    process.hltFixedGridRhoFastjetAll = cms.EDProducer( "FixedGridRhoProducerFastjet",
        gridSpacing = cms.double( 0.55 ),
        maxRapidity = cms.double( 5.0 ),
        pfCandidatesTag = cms.InputTag( "hltParticleFlowReg" )
    )

    PFTauQualityCuts.primaryVertexSrc = cms.InputTag("hltPixelVertices")

    ## Decay mode prediscriminant
    requireDecayMode = cms.PSet(
        BooleanOperator = cms.string("and"),
        decayMode = cms.PSet(
            Producer = cms.InputTag('hltHpsPFTauDiscriminationByDecayModeFindingNewDMsReg'),
            cut = cms.double(0.5)
        )
    )

    ## Cut based isolations dR=0.5
    process.hpsPFTauBasicDiscriminators = pfRecoTauDiscriminationByIsolation.clone(
        PFTauProducer = 'hltHpsPFTauProducerReg',
        Prediscriminants = requireDecayMode.clone(),
        deltaBetaPUTrackPtCutOverride     = True, # Set the boolean = True to override.
        deltaBetaPUTrackPtCutOverride_val = 0.5,  # Set the value for new value.
        particleFlowSrc = 'hltParticleFlowReg',
        vertexSrc = PFTauQualityCuts.primaryVertexSrc,
        customOuterCone = PFRecoTauPFJetInputs.isolationConeSize,
        isoConeSizeForDeltaBeta = 0.8,
        deltaBetaFactor = "%0.4f"%(ak4dBetaCorrection),
        qualityCuts = dict(isolationQualityCuts = dict(minTrackHits = 3, minGammaEt = 1.0, minTrackPt = 0.5)),
        IDdefinitions = [
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
        ],
    )

    ## Cut based isolations dR=0.3
    process.hpsPFTauBasicDiscriminatorsdR03 = process.hpsPFTauBasicDiscriminators.clone(
        customOuterCone = 0.3
    )

    process.hpsPFTauPrimaryVertexProducer = PFTauPrimaryVertexProducer.clone(
        PFTauTag = "hltHpsPFTauProducerReg",
        ElectronTag = "hltEgammaCandidates",
        MuonTag = "hltMuonsReg",
        PVTag = "hltPixelVertices",
        beamSpot = "hltOnlineBeamSpot",
        discriminators = [
            cms.PSet(
                discriminator = cms.InputTag('hltHpsPFTauDiscriminationByDecayModeFindingNewDMsReg'),
                selectionCut = cms.double(0.5)
            )
        ],
        cut = "pt > 18.0 & abs(eta) < 2.4",
        qualityCuts = PFTauQualityCuts
    )

    process.hpsPFTauSecondaryVertexProducer = PFTauSecondaryVertexProducer.clone(
        PFTauTag = "hltHpsPFTauProducerReg"
    )
    process.hpsPFTauTransverseImpactParameters = PFTauTransverseImpactParameters.clone(
        PFTauTag = "hltHpsPFTauProducerReg",
        PFTauPVATag = "hpsPFTauPrimaryVertexProducer",
        PFTauSVATag = "hpsPFTauSecondaryVertexProducer",
        useFullCalculation = True
    )

    file_names = [
    				'core:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_core.pb',
    				'inner:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_inner.pb',
    				'outer:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_outer.pb',
    			]

    working_points = ["0.", "0.92"]

    process.deepTauProducer = DeepTau.clone(
        taus = 'hltHpsPFTauProducerReg',
        pfcands = 'hltParticleFlowReg',
        vertices = 'hltPixelVertices',
        rho = 'hltFixedGridRhoFastjetAll',
        graph_file = file_names,
        disable_dxy_pca = cms.bool(True),
        is_online = cms.bool(True),
        basicTauDiscriminators = 'hpsPFTauBasicDiscriminators',
        basicTauDiscriminatorsdR03 = 'hpsPFTauBasicDiscriminatorsdR03',
        Prediscriminants = requireDecayMode.clone(),  
        VSeWP = working_points,
        VSmuWP = working_points,
        VSjetWP = working_points     
    )	

    # Add DeepTauProducer
    process.HLTHPSMediumChargedIsoPFTauSequenceReg += (process.hpsPFTauPrimaryVertexProducer  + process.hpsPFTauSecondaryVertexProducer + process.hpsPFTauTransverseImpactParameters + process.hltFixedGridRhoFastjetAll +  process.hpsPFTauBasicDiscriminators + process.hpsPFTauBasicDiscriminatorsdR03 + process.deepTauProducer)

    return process
