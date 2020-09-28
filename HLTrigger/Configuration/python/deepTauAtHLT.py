import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.PFRecoTauDiscriminationByIsolation_cfi import *
from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts
from RecoTauTag.RecoTau.PFRecoTauDiscriminationByHPSSelection_cfi import hpsSelectionDiscriminator, decayMode_1Prong0Pi0, decayMode_1Prong1Pi0, decayMode_1Prong2Pi0, decayMode_2Prong0Pi0, decayMode_2Prong1Pi0, decayMode_3Prong0Pi0, decayMode_3Prong1Pi0

from RecoTauTag.RecoTau.PFTauPrimaryVertexProducer_cfi      import *
from RecoTauTag.RecoTau.PFTauSecondaryVertexProducer_cfi    import *
from RecoTauTag.RecoTau.PFTauTransverseImpactParameters_cfi import *

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
        PFTauProducer = cms.InputTag('hltHpsPFTauProducerReg'),
        Prediscriminants = requireDecayMode.clone(),
        # Prediscriminants = cms.PSet(  BooleanOperator = cms.string( "and" ) ),
        particleFlowSrc = cms.InputTag("hltParticleFlowReg"),
        vertexSrc = PFTauQualityCuts.primaryVertexSrc,
        customOuterCone = 0.5,
        isoConeSizeForDeltaBeta = 0.8,
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
                applyDeltaBetaCorrection = cms.bool(True),
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
    )

    ## Cut based isolations dR=0.3
    process.hpsPFTauBasicDiscriminatorsdR03 = process.hpsPFTauBasicDiscriminators.clone(
        customOuterCone = cms.double(0.3)
    )

    process.hpsPFTauPrimaryVertexProducer = PFTauPrimaryVertexProducer.clone(
        PFTauTag = cms.InputTag("hltHpsPFTauProducerReg"),
        ElectronTag = cms.InputTag("hltEgammaCandidates"),
        MuonTag = cms.InputTag("hltMuonsReg"),
        PVTag = cms.InputTag("hltPixelVertices"),
        beamSpot = cms.InputTag("hltOnlineBeamSpot"),
        Algorithm = cms.int32(0),
        useBeamSpot = True,
        RemoveMuonTracks = False,
        RemoveElectronTracks = False,
        useSelectedTaus = False,
        discriminators = cms.VPSet(
            cms.PSet(
                discriminator = cms.InputTag('hltHpsPFTauDiscriminationByDecayModeFindingNewDMsReg'),
                selectionCut = cms.double(0.5)
            )
        ),
        cut = cms.string("pt > 18.0 & abs(eta) < 2.4"),
        qualityCuts = cms.PSet(
            isolationQualityCuts = cms.PSet(
                maxDeltaZ = cms.double(0.2),
                maxTrackChi2 = cms.double(100.0),
                maxTransverseImpactParameter = cms.double(0.03),
                minGammaEt = cms.double(1.5),
                minTrackHits = cms.uint32(8),
                minTrackPixelHits = cms.uint32(0),
                minTrackPt = cms.double(1.0),
                minTrackVertexWeight = cms.double(-1.0)
            ),
            leadingTrkOrPFCandOption = cms.string('leadPFCand'),
            primaryVertexSrc = cms.InputTag("hltPixelVertices"),
            pvFindingAlgo = cms.string('closestInDeltaZ'),
            recoverLeadingTrk = cms.bool(False),
            signalQualityCuts = cms.PSet(
                maxDeltaZ = cms.double(0.4),
                maxTrackChi2 = cms.double(100.0),
                maxTransverseImpactParameter = cms.double(0.1),
                minGammaEt = cms.double(1.0),
                minNeutralHadronEt = cms.double(30.0),
                minTrackHits = cms.uint32(3),
                minTrackPixelHits = cms.uint32(0),
                minTrackPt = cms.double(0.5),
                minTrackVertexWeight = cms.double(-1.0)
            ),
            vertexTrackFiltering = cms.bool(False),
            vxAssocQualityCuts = cms.PSet(
                maxTrackChi2 = cms.double(100.0),
                maxTransverseImpactParameter = cms.double(0.1),
                minGammaEt = cms.double(1.0),
                minTrackHits = cms.uint32(3),
                minTrackPixelHits = cms.uint32(0),
                minTrackPt = cms.double(0.5),
                minTrackVertexWeight = cms.double(-1.0)
            )
        )
    )

    process.hpsPFTauSecondaryVertexProducer = PFTauSecondaryVertexProducer.clone(
        PFTauTag = cms.InputTag("hltHpsPFTauProducerReg")
    )
    process.hpsPFTauTransverseImpactParameters = PFTauTransverseImpactParameters.clone(
        PFTauTag = cms.InputTag("hltHpsPFTauProducerReg"),
        PFTauPVATag = cms.InputTag("hpsPFTauPrimaryVertexProducer"),
        PFTauSVATag = cms.InputTag("hpsPFTauSecondaryVertexProducer"),
        useFullCalculation = True
    )


    #Add all sums and corrections for deepTauProducer
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.hpsPFTauPrimaryVertexProducer)
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.hpsPFTauSecondaryVertexProducer)
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.hpsPFTauTransverseImpactParameters)
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.hltFixedGridRhoFastjetAll)
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.hpsPFTauBasicDiscriminators)
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.hpsPFTauBasicDiscriminatorsdR03)

    file_names = [
    				'core:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_core.pb',
    				'inner:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_inner.pb',
    				'outer:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_outer.pb',
    			]

    wp_names = ["0.", "0."]

    process.deepTauProducer = cms.EDProducer("DeepTauId",
    taus                                = cms.InputTag("hltHpsPFTauProducerReg"),
    pfcands                             = cms.InputTag('hltParticleFlowReg'),
    vertices                            = cms.InputTag('hltPixelVertices'),
    rho                                 = cms.InputTag('hltFixedGridRhoFastjetAll'),
    graph_file                          = cms.vstring(file_names),
    mem_mapped                          = cms.bool(False),
    version                             = cms.uint32(2),
    debug_level                         = cms.int32(0),
    disable_dxy_pca                     = cms.bool(True),
    is_online                 		    = cms.bool(True),
    pfTauTransverseImpactParameters     = cms.InputTag('hpsPFTauTransverseImpactParameters'),
    chargedIsoPtSum_index               = cms.uint32(0),
    neutralIsoPtSum_index               = cms.uint32(1),
    puCorrPtSum_index                   = cms.uint32(5),
    tauFootPrintCorrection_index        = cms.uint32(3),
    neutralIsoPtSumWeight_index         = cms.uint32(2),
    photonPtSumOutsideSignalCone_index  = cms.uint32(4),
    basicTauDiscriminators              = cms.InputTag('hpsPFTauBasicDiscriminators'),
    basicTauDiscriminatorsdR03          = cms.InputTag('hpsPFTauBasicDiscriminatorsdR03'),
    Prediscriminants = requireDecayMode.clone(),

    VSeWP = cms.vstring(wp_names),
    VSmuWP = cms.vstring(wp_names),
    VSjetWP = cms.vstring(wp_names)
    )		

    #Add DeepTauProducer
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.deepTauProducer)

    return process
