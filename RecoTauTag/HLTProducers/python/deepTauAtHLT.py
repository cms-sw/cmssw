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

    process.hltFixedGridRhoFastjetAllTau = cms.EDProducer( "FixedGridRhoProducerFastjet",
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

    # process.genFilter = cms.EDFilter("GenTauFilter",
    #     genParticles         = cms.InputTag('genParticles'),
    #     n_tau_had            = cms.int32(2),
    #     n_tau_mu             = cms.int32(0),
    #     n_tau_e              = cms.int32(0)
    # )

    # process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4.insert(0, process.genFilter)

    process.genCounter = cms.EDFilter( "CounterFilter",
        isMC = cms.bool(True), #from outside
        store_hist = cms.bool(False), #from outside
        store_both = cms.bool(True), #from outside
        position = cms.string("gen"),
        deepTauVSe = cms.InputTag('try1'),
        deepTauVSmu = cms.InputTag('try2'),
        deepTauVSjet = cms.InputTag('try3'),
        isoAbs = cms.InputTag('try4'),
        isoRel = cms.InputTag('try5'),
        original_taus = cms.InputTag('try6'),
        taus = cms.InputTag('try14'),
        puInfo = cms.InputTag('try7'),
        vertices = cms.InputTag('try8'),
        decayModeFindingNewDM = cms.InputTag('try9'),
        genParticles = cms.InputTag('genParticles')
    )

    process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4.insert(1, process.genCounter)

    ## Cut based isolations dR=0.5
<<<<<<< HEAD
    process.hpsPFTauBasicDiscriminatorsForDeepTau = pfRecoTauDiscriminationByIsolation.clone(
        PFTauProducer = 'hltHpsL1JetsHLTForDeepTauInput',
=======
    process.hpsPFTauBasicDiscriminators = pfRecoTauDiscriminationByIsolation.clone(
        PFTauProducer = 'hltHpsPFTauProducerReg',
>>>>>>> Switch to CMSSW_11_3_X
        Prediscriminants = cms.PSet(  BooleanOperator = cms.string( "and" ) ),
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
    process.hpsPFTauBasicDiscriminatorsdR03ForDeepTau = process.hpsPFTauBasicDiscriminatorsForDeepTau.clone(
        customOuterCone = 0.3
    )

    process.hpsPFTauPrimaryVertexProducerForDeepTau = PFTauPrimaryVertexProducer.clone(
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

    process.hpsPFTauSecondaryVertexProducerForDeepTau = PFTauSecondaryVertexProducer.clone(
        PFTauTag = "hltHpsPFTauProducerReg"
    )
    process.hpsPFTauTransverseImpactParametersForDeepTau = PFTauTransverseImpactParameters.clone(
        PFTauTag = "hltHpsPFTauProducerReg",
        PFTauPVATag = "hpsPFTauPrimaryVertexProducerForDeepTau",
        PFTauSVATag = "hpsPFTauSecondaryVertexProducerForDeepTau",
        useFullCalculation = True
    )

    chargedIsolationQualityCuts = PFTauQualityCuts.clone(
        isolationQualityCuts = cms.PSet( 
            maxDeltaZ = cms.double( 0.2 ),
            minTrackPt = cms.double( 0.5 ),
            minGammaEt = cms.double( 0.5 ),
            minTrackHits = cms.uint32( 3 ),
            minTrackPixelHits = cms.uint32( 0 ),
            maxTrackChi2 = cms.double( 100.0 ),
            maxTransverseImpactParameter = cms.double( 0.1 ),
            useTracksInsteadOfPFHadrons = cms.bool( False )
        ),
        primaryVertexSrc = cms.InputTag( "hltPixelVertices" ),
        signalQualityCuts = cms.PSet( 
            maxDeltaZ = cms.double( 0.2 ),
            minTrackPt = cms.double( 0.0 ),
            minGammaEt = cms.double( 0.5 ),
            minTrackHits = cms.uint32( 3 ),
            minTrackPixelHits = cms.uint32( 0 ),
            maxTrackChi2 = cms.double( 1000.0 ),
            maxTransverseImpactParameter = cms.double( 0.2 ),
            useTracksInsteadOfPFHadrons = cms.bool( False ),
            minNeutralHadronEt = cms.double( 1.0 )
        ),
        vxAssocQualityCuts = cms.PSet( 
            minTrackPt = cms.double( 0.0 ),
            minGammaEt = cms.double( 0.5 ),
            minTrackHits = cms.uint32( 3 ),
            minTrackPixelHits = cms.uint32( 0 ),
            maxTrackChi2 = cms.double( 1000.0 ),
            maxTransverseImpactParameter = cms.double( 0.2 ),
            useTracksInsteadOfPFHadrons = cms.bool( False )
        ),
    )

<<<<<<< HEAD
    process.hltHpsL1JetsHLTForDeepTauInput = process.hltHpsL1JetsHLTDoublePFTauTrackPt1MediumChargedIsolationMatchReg.clone(
        JetSrc = cms.InputTag('hltHpsPFTauProducerReg'),
        ReduceTauContent = cms.bool(False),
        KeepOriginalVertex = cms.bool(True),
=======
    process.hpsPFTauAbsoluteChargedIsolationDiscriminator = pfRecoTauDiscriminationByIsolation.clone(
        PFTauProducer = 'hltHpsPFTauProducerReg',
        vertexSrc = "NotUsed",
        rhoProducer = "hltFixedGridRhoFastjetAll",
        Prediscriminants = cms.PSet(  BooleanOperator = cms.string( "and" ) ),
        WeightECALIsolation = 0.333,
        deltaBetaPUTrackPtCutOverride     = False,
        deltaBetaPUTrackPtCutOverride_val = 0.5,
        isoConeSizeForDeltaBeta = 0.3,
        qualityCuts = chargedIsolationQualityCuts,
        IDdefinitions = [
            cms.PSet(
                IDname = cms.string("AbsoluteChargedIsolation"),
                ApplyDiscriminationByTrackerIsolation = cms.bool(True),
                storeRawSumPt = cms.bool(True)
            )
        ],
        IDWPdefinitions = [
            cms.PSet(
                IDname = cms.string("LooseChargedIsolation"),
                referenceRawIDNames = cms.vstring("AbsoluteChargedIsolation"),
                maximumAbsoluteValues = cms.vdouble(3.9),
                ),
            cms.PSet(
                IDname = cms.string("MediumChargedIsolation"),
                referenceRawIDNames = cms.vstring("AbsoluteChargedIsolation"),
                maximumAbsoluteValues = cms.vdouble(3.7),
                ),
            cms.PSet(
                IDname = cms.string("TightChargedIsolation"),
                referenceRawIDNames = cms.vstring("AbsoluteChargedIsolation"),
                maximumAbsoluteValues = cms.vdouble(3.2),
                ),
        ]
    )

    process.hpsPFTauRelativeChargedIsolationDiscriminator = pfRecoTauDiscriminationByIsolation.clone(
        PFTauProducer = 'hltHpsPFTauProducerReg',
        vertexSrc = "NotUsed",
        applyRhoCorrection = True,
        rhoProducer = "hltFixedGridRhoFastjetAll",
        Prediscriminants = cms.PSet(  BooleanOperator = cms.string( "and" ) ),
        WeightECALIsolation = 1.0,
        deltaBetaPUTrackPtCutOverride     = False,
        deltaBetaPUTrackPtCutOverride_val = 0.5,
        isoConeSizeForDeltaBeta = 0.3,
        qualityCuts = chargedIsolationQualityCuts,
        IDdefinitions = [
            cms.PSet(
                IDname = cms.string("RelativeChargedIsolation"),
                ApplyDiscriminationByTrackerIsolation = cms.bool(True),
                storeRawSumPt = cms.bool(True)
            )
        ],
        IDWPdefinitions = [
            cms.PSet(
                IDname = cms.string("LooseChargedIsolation"),
                referenceRawIDNames = cms.vstring("RelativeChargedIsolation"),
                maximumRelativeValues = cms.vdouble(0.05),
                relativeValueOffsets = cms.vdouble(50.)
                ),
            cms.PSet(
                IDname = cms.string("MediumChargedIsolation"),
                referenceRawIDNames = cms.vstring("RelativeChargedIsolation"),
                maximumRelativeValues = cms.vdouble(0.05),
                relativeValueOffsets = cms.vdouble(60.)
                ),
            cms.PSet(
                IDname = cms.string("TightChargedIsolation"),
                referenceRawIDNames = cms.vstring("RelativeChargedIsolation"),
                maximumRelativeValues = cms.vdouble(0.04),
                relativeValueOffsets = cms.vdouble(70.)
                ),
        ]
>>>>>>> Switch to CMSSW_11_3_X
    )

    file_names = [
    				'core:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_core.pb',
    				'inner:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_inner.pb',
    				'outer:RecoTauTag/TrainingFiles/data/DeepTauId/deepTau_2017v2p6_e6_outer.pb',
    			]

    def getLinExpression(x1, x2, y1, y2):
        return "(((({3}-{2})/({1}-{0}))*(pt-{0}))+{2})".format(x1, x2, y1, y2)

    val1, val2 = ("0.49948551", "0.125")
    working_points = ["return {0}*(pt < 35)+".format(val1)+getLinExpression("35", "300", val1, val2)+ "*(35 <= pt && pt < 300) + {0}*(pt >= 300);".format(val2)]
    working_points = ["return 0.49948551*(pt < 35)+(-0.00141315*(pt-35)+0.49948551)*(35 <= pt && pt < 300) + 0.125*(pt >= 300);"]

    process.deepTauProducer = DeepTau.clone(
        taus = 'hltHpsL1JetsHLTForDeepTauInput',
        # taus = 'hltHpsPFTauProducerReg',
        pfcands = 'hltParticleFlowReg',
        vertices = 'hltPixelVertices',
        rho = 'hltFixedGridRhoFastjetAllTau',
        graph_file = file_names,
        disable_dxy_pca = cms.bool(True),
        is_online = cms.bool(True),
<<<<<<< HEAD
        basicTauDiscriminators = 'hpsPFTauBasicDiscriminatorsForDeepTau',
        basicTauDiscriminatorsdR03 = 'hpsPFTauBasicDiscriminatorsdR03ForDeepTau',
        pfTauTransverseImpactParameters = 'hpsPFTauTransverseImpactParametersForDeepTau',
=======
        basicTauDiscriminators = 'hpsPFTauBasicDiscriminators',
        basicTauDiscriminatorsdR03 = 'hpsPFTauBasicDiscriminatorsdR03',
>>>>>>> Switch to CMSSW_11_3_X
        Prediscriminants = cms.PSet(  BooleanOperator = cms.string( "and" ) ),  
        VSeWP = working_points,
        VSmuWP = working_points,
        VSjetWP = working_points     
    )	

<<<<<<< HEAD
    # Add DeepTauProducer
    process.HLTHPSDeepTau35IsoPFTauSequenceReg = cms.Sequence(process.hpsPFTauPrimaryVertexProducerForDeepTau + process.hpsPFTauSecondaryVertexProducerForDeepTau + process.hpsPFTauTransverseImpactParametersForDeepTau + process.hltFixedGridRhoFastjetAllTau + process.hltHpsL1JetsHLTForDeepTauInput + process.hpsPFTauBasicDiscriminatorsForDeepTau + process.hpsPFTauBasicDiscriminatorsdR03ForDeepTau + process.deepTauProducer)
    process.hltHpsSelectedPFTausTrackPt1DeepTau35IsolationReg = process.hltHpsSelectedPFTausTrackPt1MediumChargedIsolationReg.clone(
        discriminators = [],
        # discriminators = [
        #     cms.PSet(  
        #         discriminator = cms.InputTag( "hltHpsPFTauTrackPt1DiscriminatorReg" ),
        #         selectionCut = cms.double( 0.5 )
        #     )
        # ],
        src = 'hltHpsL1JetsHLTForDeepTauInput',
        discriminatorContainers = [
            cms.PSet(  
                discriminator = cms.InputTag( "deepTauProducer", "VSjet" ),
                rawValues = cms.vstring(),
                selectionCuts = cms.vdouble(),
                workingPoints = cms.vstring(working_points),
            )
        ]
    )

    process.hltHpsDoublePFTau35TrackPt1DeepTau35IsolationReg = process.hltHpsDoublePFTau35TrackPt1MediumChargedIsolationReg.clone(
        inputTag = "hltHpsSelectedPFTausTrackPt1DeepTau35IsolationReg",
    )

    process.hltHpsL1JetsHLTDoublePFTauTrackPt1DeepTauMatchReg = process.hltHpsL1JetsHLTDoublePFTauTrackPt1MediumChargedIsolationMatchReg.clone(
        JetSrc = "hltHpsSelectedPFTausTrackPt1DeepTau35IsolationReg",
    )

    process.hltHpsDoublePFTau35TrackPt1DeepTauL1HLTMatchedReg = process.hltHpsDoublePFTau35TrackPt1MediumChargedIsolationL1HLTMatchedReg.clone(
        inputTag = "hltHpsL1JetsHLTDoublePFTauTrackPt1DeepTauMatchReg",
    )

    process.hltHpsDoublePFTau35TrackPt1DeepTau35IsolationDz02Reg = process.hltHpsDoublePFTau35TrackPt1MediumChargedIsolationDz02Reg.clone(
        JetSrc = "hltHpsL1JetsHLTDoublePFTauTrackPt1DeepTauMatchReg"
    )

    process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4.remove(process.HLTHPSMediumChargedIsoPFTauSequenceReg)
    process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4.remove(process.hltHpsSelectedPFTausTrackPt1MediumChargedIsolationReg)
    process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4.remove(process.hltHpsDoublePFTau35TrackPt1MediumChargedIsolationReg)
    process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4.remove(process.hltHpsL1JetsHLTDoublePFTauTrackPt1MediumChargedIsolationMatchReg)
    process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4.remove(process.hltHpsDoublePFTau35TrackPt1MediumChargedIsolationL1HLTMatchedReg)
    process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4.remove(process.hltHpsDoublePFTau35TrackPt1MediumChargedIsolationDz02Reg)
    process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4.remove(process.HLTEndSequence)
    
    process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4 += (process.HLTHPSDeepTau35IsoPFTauSequenceReg + process.hltHpsSelectedPFTausTrackPt1DeepTau35IsolationReg + process.hltHpsDoublePFTau35TrackPt1DeepTau35IsolationReg + process.hltHpsL1JetsHLTDoublePFTauTrackPt1DeepTauMatchReg + process.hltHpsDoublePFTau35TrackPt1DeepTauL1HLTMatchedReg + process.hltHpsDoublePFTau35TrackPt1DeepTau35IsolationDz02Reg + process.HLTEndSequence)
=======
    process.jetsFilter = cms.EDFilter( "CounterFilter",
        isMC = cms.bool(True), #from outside
        store_hist = cms.bool(False), #from outside
        store_both = cms.bool(False), #from outside
        position = cms.string("initial"),
        deepTauVSe = cms.InputTag('deepTauProducer', 'VSe'),
        deepTauVSmu = cms.InputTag('deepTauProducer', 'VSmu'),
        deepTauVSjet = cms.InputTag('deepTauProducer', 'VSjet'),
        isoAbs = cms.InputTag('hpsPFTauAbsoluteChargedIsolationDiscriminator'),
        isoRel = cms.InputTag('hpsPFTauRelativeChargedIsolationDiscriminator'),
        original_taus = cms.InputTag('hltHpsPFTauProducerReg'),
        taus = cms.InputTag('hltHpsPFTauProducerReg'),
        puInfo = cms.InputTag('addPileupInfo','','HLT'),
        vertices = cms.InputTag('hltPixelVertices'),
        decayModeFindingNewDM = cms.InputTag('hltHpsPFTauDiscriminationByDecayModeFindingNewDMsReg'),
        genParticles = cms.InputTag('genParticles')
    )
    process.TFileService = cms.Service("TFileService", fileName = cms.string("histo.root"))
    process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4.insert(-1, process.jetsFilter)
>>>>>>> Switch to CMSSW_11_3_X


    # Add DeepTauProducer
    process.HLTHPSMediumChargedIsoPFTauSequenceReg += (process.hpsPFTauPrimaryVertexProducer  + process.hpsPFTauSecondaryVertexProducer + process.hpsPFTauTransverseImpactParameters + process.hltFixedGridRhoFastjetAll + process.hpsPFTauAbsoluteChargedIsolationDiscriminator + process.hpsPFTauRelativeChargedIsolationDiscriminator + process.hpsPFTauBasicDiscriminators + process.hpsPFTauBasicDiscriminatorsdR03 + process.deepTauProducer)
    process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4.remove(process.hltHpsSelectedPFTausTrackPt1MediumChargedIsolationReg)
    process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4.remove(process.hltHpsDoublePFTau35TrackPt1MediumChargedIsolationReg)
    process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4.remove(process.hltHpsL1JetsHLTDoublePFTauTrackPt1MediumChargedIsolationMatchReg)
    process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4.remove(process.hltHpsDoublePFTau35TrackPt1MediumChargedIsolationL1HLTMatchedReg)
    process.HLT_DoubleMediumChargedIsoPFTauHPS35_Trk1_eta2p1_Reg_v4.remove(process.hltHpsDoublePFTau35TrackPt1MediumChargedIsolationDz02Reg)
    return process
