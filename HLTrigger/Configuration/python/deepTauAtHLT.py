import FWCore.ParameterSet.Config as cms

from RecoTauTag.RecoTau.pfRecoTauDiscriminationByIsolation_cfi import *
from RecoTauTag.RecoTau.PFRecoTauQualityCuts_cfi import PFTauQualityCuts


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

    process.chargedIsoPtSum = pfRecoTauDiscriminationByIsolation.clone(
        PFTauProducer = cms.InputTag('hltHpsPFTauProducerReg'),
        particleFlowSrc = cms.InputTag("hltParticleFlowReg"),
        vertexSrc = PFTauQualityCuts.primaryVertexSrc,
        Prediscriminants = cms.PSet(  BooleanOperator = cms.string( "and" ) ),
        ApplyDiscriminationByECALIsolation = cms.bool(False),
        ApplyDiscriminationByTrackerIsolation = cms.bool(True),
        applyDeltaBetaCorrection = cms.bool(False),
        storeRawSumPt = cms.bool(True),
        applyOccupancyCut = cms.bool(False),
        applySumPtCut = cms.bool(False),
        storeRawPUsumPt = cms.bool(False),
        customOuterCone = cms.double(0.5),
        isoConeSizeForDeltaBeta = cms.double(0.8),
        verbosity = cms.int32(0),
        qualityCuts = PFTauQualityCuts
    )

    process.chargedIsoPtSumdR03 = pfRecoTauDiscriminationByIsolation.clone(
        PFTauProducer = cms.InputTag('hltHpsPFTauProducerReg'),
        particleFlowSrc = cms.InputTag("hltParticleFlowReg"),
        vertexSrc = PFTauQualityCuts.primaryVertexSrc,
        ApplyDiscriminationByECALIsolation = cms.bool(False),
        ApplyDiscriminationByTrackerIsolation = cms.bool(True),
        Prediscriminants = cms.PSet(  BooleanOperator = cms.string( "and" ) ),
        applyOccupancyCut = cms.bool(False),
        applySumPtCut = cms.bool(False),
        applyDeltaBetaCorrection = cms.bool(False),
        storeRawSumPt = cms.bool(True),
        storeRawPUsumPt = cms.bool(False),
        customOuterCone = cms.double(0.3),
        isoConeSizeForDeltaBeta = cms.double(0.8),
        verbosity = cms.int32(0),
        qualityCuts = PFTauQualityCuts
    )
    process.neutralIsoPtSum = pfRecoTauDiscriminationByIsolation.clone(
        PFTauProducer = cms.InputTag('hltHpsPFTauProducerReg'),
        particleFlowSrc = cms.InputTag("hltParticleFlowReg"),
        vertexSrc = cms.InputTag("hltPixelVertices"),
        ApplyDiscriminationByECALIsolation = cms.bool(True),
        ApplyDiscriminationByTrackerIsolation = cms.bool(False),
        Prediscriminants = cms.PSet(  BooleanOperator = cms.string( "and" ) ),
        applyOccupancyCut = cms.bool(False),
        applySumPtCut = cms.bool(False),
        applyDeltaBetaCorrection = cms.bool(False),
        storeRawSumPt = cms.bool(True),
        storeRawPUsumPt = cms.bool(False),
        customOuterCone = cms.double(0.5),
        isoConeSizeForDeltaBeta = cms.double(0.8),
        verbosity = cms.int32(0),
        qualityCuts = PFTauQualityCuts
    )
    process.neutralIsoPtSumdR03 = pfRecoTauDiscriminationByIsolation.clone(
        PFTauProducer = cms.InputTag('hltHpsPFTauProducerReg'),
        particleFlowSrc = cms.InputTag("hltParticleFlowReg"),
        vertexSrc = cms.InputTag("hltPixelVertices"),
        ApplyDiscriminationByECALIsolation = cms.bool(True),
        ApplyDiscriminationByTrackerIsolation = cms.bool(False),
        Prediscriminants = cms.PSet(  BooleanOperator = cms.string( "and" ) ),
        applyOccupancyCut = cms.bool(False),
        applySumPtCut = cms.bool(False),
        applyDeltaBetaCorrection = cms.bool(False),
        storeRawSumPt = cms.bool(True),
        storeRawPUsumPt = cms.bool(False),
        customOuterCone = cms.double(0.3),
        isoConeSizeForDeltaBeta = cms.double(0.8),
        verbosity = cms.int32(0),
        qualityCuts = PFTauQualityCuts
    )
    process.puCorrPtSum = pfRecoTauDiscriminationByIsolation.clone(
        PFTauProducer = cms.InputTag('hltHpsPFTauProducerReg'),
        particleFlowSrc = cms.InputTag("hltParticleFlowReg"),
        vertexSrc = cms.InputTag("hltPixelVertices"),
        ApplyDiscriminationByECALIsolation = cms.bool(False),
        ApplyDiscriminationByTrackerIsolation = cms.bool(False),
        Prediscriminants = cms.PSet(  BooleanOperator = cms.string( "and" ) ),
        applyOccupancyCut = cms.bool(False),
        applySumPtCut = cms.bool(False),
        applyDeltaBetaCorrection = cms.bool(True),
        storeRawSumPt = cms.bool(False),
        storeRawPUsumPt = cms.bool(True),
        customOuterCone = cms.double(0.5),
        isoConeSizeForDeltaBeta = cms.double(0.8),
        verbosity = cms.int32(0),
        qualityCuts = PFTauQualityCuts
    )
    requireDecayMode = cms.PSet(
        BooleanOperator = cms.string("and"),
        decayMode = cms.PSet(
            Producer = cms.InputTag('hltHpsPFTauDiscriminationByDecayModeFindingNewDMsReg'),
            cut = cms.double(0.5)
        )
    )
    process.footprintCorrection = pfRecoTauDiscriminationByIsolation.clone(
        PFTauProducer = cms.InputTag('hltHpsPFTauProducerReg'),
        particleFlowSrc = cms.InputTag("hltParticleFlowReg"),
        vertexSrc = cms.InputTag("hltPixelVertices"),
        Prediscriminants = requireDecayMode.clone(),
        ApplyDiscriminationByECALIsolation = cms.bool(False),
        ApplyDiscriminationByTrackerIsolation = cms.bool(False),
        applyOccupancyCut = cms.bool(False),
        applySumPtCut = cms.bool(False),
        applyDeltaBetaCorrection = cms.bool(True),
        storeRawSumPt = cms.bool(False),
        storeRawPUsumPt = cms.bool(False),
        storeRawFootprintCorrection = cms.bool(True),
        customOuterCone = cms.double(0.5),
        isoConeSizeForDeltaBeta = cms.double(0.8),
        verbosity = cms.int32(0),
        qualityCuts = PFTauQualityCuts
    )
    process.footprintCorrectiondR03 = pfRecoTauDiscriminationByIsolation.clone(
        PFTauProducer = cms.InputTag('hltHpsPFTauProducerReg'),
        particleFlowSrc = cms.InputTag("hltParticleFlowReg"),
        vertexSrc = cms.InputTag("hltPixelVertices"),
        Prediscriminants = requireDecayMode.clone(),
        ApplyDiscriminationByECALIsolation = cms.bool(False),
        ApplyDiscriminationByTrackerIsolation = cms.bool(False),
        applyOccupancyCut = cms.bool(False),
        applySumPtCut = cms.bool(False),
        applyDeltaBetaCorrection = cms.bool(True),
        storeRawSumPt = cms.bool(False),
        storeRawPUsumPt = cms.bool(False),
        storeRawFootprintCorrection = cms.bool(True),
        customOuterCone = cms.double(0.3),
        isoConeSizeForDeltaBeta = cms.double(0.8),
        verbosity = cms.int32(0),
        qualityCuts = PFTauQualityCuts
    )

    process.neutralIsoPtSumWeight = pfRecoTauDiscriminationByIsolation.clone(
        PFTauProducer = cms.InputTag('hltHpsPFTauProducerReg'),
        particleFlowSrc = cms.InputTag("hltParticleFlowReg"),
        vertexSrc = cms.InputTag("hltPixelVertices"),
        Prediscriminants = requireDecayMode.clone(),
        ApplyDiscriminationByECALIsolation = cms.bool(True),
        ApplyDiscriminationByTrackerIsolation = cms.bool(False),
        UseAllPFCandsForWeights = cms.bool(True),
        applyOccupancyCut = cms.bool(False),
        applySumPtCut = cms.bool(False),
        applyDeltaBetaCorrection = cms.bool(False),
        storeRawSumPt = cms.bool(True),
        storeRawPUsumPt = cms.bool(False),
        customOuterCone = cms.double(0.5),
        isoConeSizeForDeltaBeta = cms.double(0.8),
        verbosity = cms.int32(0),
        qualityCuts = PFTauQualityCuts
    )

    process.neutralIsoPtSumWeightdR03 = pfRecoTauDiscriminationByIsolation.clone(
        PFTauProducer = cms.InputTag('hltHpsPFTauProducerReg'),
        particleFlowSrc = cms.InputTag("hltParticleFlowReg"),
        vertexSrc = cms.InputTag("hltPixelVertices"),
        Prediscriminants = requireDecayMode.clone(),
        ApplyDiscriminationByECALIsolation = cms.bool(True),
        ApplyDiscriminationByTrackerIsolation = cms.bool(False),
        UseAllPFCandsForWeights = cms.bool(True),
        applyOccupancyCut = cms.bool(False),
        applySumPtCut = cms.bool(False),
        applyDeltaBetaCorrection = cms.bool(False),
        storeRawSumPt = cms.bool(True),
        storeRawPUsumPt = cms.bool(False),
        customOuterCone = cms.double(0.5),
        isoConeSizeForDeltaBeta = cms.double(0.8),
        verbosity = cms.int32(0),
        qualityCuts = PFTauQualityCuts
    )

    process.photonPtSumOutsideSignalCone = pfRecoTauDiscriminationByIsolation.clone(
        PFTauProducer = cms.InputTag('hltHpsPFTauProducerReg'),
        particleFlowSrc = cms.InputTag("hltParticleFlowReg"),
        vertexSrc = cms.InputTag("hltPixelVertices"),
        Prediscriminants = requireDecayMode.clone(),
        ApplyDiscriminationByECALIsolation = cms.bool(False),
        ApplyDiscriminationByTrackerIsolation = cms.bool(False),
        applyOccupancyCut = cms.bool(False),
        applySumPtCut = cms.bool(False),
        applyDeltaBetaCorrection = cms.bool(False),
        storeRawSumPt = cms.bool(False),
        storeRawPUsumPt = cms.bool(False),
        storeRawPhotonSumPt_outsideSignalCone = cms.bool(True),
        customOuterCone = cms.double(0.5),
        isoConeSizeForDeltaBeta = cms.double(0.8),
        verbosity = cms.int32(0),
        qualityCuts = PFTauQualityCuts
    )
    process.photonPtSumOutsideSignalConedR03 = pfRecoTauDiscriminationByIsolation.clone(
        PFTauProducer = cms.InputTag('hltHpsPFTauProducerReg'),
        particleFlowSrc = cms.InputTag("hltParticleFlowReg"),
        vertexSrc = cms.InputTag("hltPixelVertices"),
        Prediscriminants = requireDecayMode.clone(),
        ApplyDiscriminationByECALIsolation = cms.bool(False),
        ApplyDiscriminationByTrackerIsolation = cms.bool(False),
        applyOccupancyCut = cms.bool(False),
        applySumPtCut = cms.bool(False),
        applyDeltaBetaCorrection = cms.bool(False),
        storeRawSumPt = cms.bool(False),
        storeRawPUsumPt = cms.bool(False),
        storeRawPhotonSumPt_outsideSignalCone = cms.bool(True),
        customOuterCone = cms.double(0.3),
        isoConeSizeForDeltaBeta = cms.double(0.8),
        verbosity = cms.int32(0),
        qualityCuts = PFTauQualityCuts
    )

    process.hpsPFTauPrimaryVertexProducer = PFTauPrimaryVertexProducer.clone(
        PFTauTag = cms.InputTag("hltHpsPFTauProducerReg"),
        ElectronTag = cms.InputTag("hltEgammaCandidates"),
        MuonTag = cms.InputTag("hltMuonsReg"),
        PVTag = cms.InputTag("hltPixelVertices"),
        beamSpot = cms.InputTag("hltOnlineBeamSpot"),
        Algorithm = cms.int32(0),
        useBeamSpot = cms.bool(True),
        RemoveMuonTracks = cms.bool(False),
        RemoveElectronTracks = cms.bool(False),
        useSelectedTaus = cms.bool(False),
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
        useFullCalculation = cms.bool(True)
    )


    #Add all sums and corrections for deepTauProducer
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.hpsPFTauPrimaryVertexProducer)
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.hpsPFTauSecondaryVertexProducer)
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.hpsPFTauTransverseImpactParameters)
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.chargedIsoPtSum)
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.chargedIsoPtSumdR03)
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.neutralIsoPtSum)
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.neutralIsoPtSumdR03)
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.puCorrPtSum)
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.footprintCorrection)
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.footprintCorrectiondR03)
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.neutralIsoPtSumWeight)
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.neutralIsoPtSumWeightdR03)
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.photonPtSumOutsideSignalCone)
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.photonPtSumOutsideSignalConedR03)
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.hltFixedGridRhoFastjetAll)


    file_names = [
    				'core:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_core.pb',
    				'inner:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_inner.pb',
    				'outer:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_outer.pb',
    			]

    wp_names = ["0.", "0."]

    process.deepTauProducer = cms.EDProducer("DeepTauId",
    taus                            = cms.InputTag("hltHpsPFTauProducerReg"),
    pfcands                         = cms.InputTag('hltParticleFlowReg'),
    vertices                        = cms.InputTag('hltPixelVertices'),
    rho                             = cms.InputTag('hltFixedGridRhoFastjetAll'),
    graph_file                      = cms.vstring(file_names),
    mem_mapped                      = cms.bool(False),
    version                         = cms.uint32(2),
    debug_level                     = cms.int32(0),
    disable_dxy_pca                 = cms.bool(True),
    is_online                 		  = cms.bool(True),
    pfTauTransverseImpactParameters = cms.InputTag('hpsPFTauTransverseImpactParameters'),
    chargedIsoPtSum                 = cms.InputTag('chargedIsoPtSum'),
    chargedIsoPtSumdR03             = cms.InputTag('chargedIsoPtSumdR03'),
    neutralIsoPtSum                 = cms.InputTag('neutralIsoPtSum'),
    neutralIsoPtSumdR03             = cms.InputTag('neutralIsoPtSumdR03'),
    puCorrPtSum                     = cms.InputTag('puCorrPtSum'),
    footprintCorrection             = cms.InputTag('footprintCorrection'),
    neutralIsoPtSumWeight           = cms.InputTag('neutralIsoPtSumWeight'),
    neutralIsoPtSumWeightdR03       = cms.InputTag('neutralIsoPtSumWeightdR03'),
    photonPtSumOutsideSignalCone    = cms.InputTag('photonPtSumOutsideSignalCone'),
    photonPtSumOutsideSignalConedR03 = cms.InputTag('photonPtSumOutsideSignalConedR03'),

    VSeWP = cms.vstring(wp_names),
    VSmuWP = cms.vstring(wp_names),
    VSjetWP = cms.vstring(wp_names)
    )		

    #Add DeepTauProducer
    process.HLTHPSMediumChargedIsoPFTauSequenceReg.insert(-1, process.deepTauProducer)
 
    process.maxEvents = cms.untracked.PSet(
        input = cms.untracked.int32(200)
    )

    return process
