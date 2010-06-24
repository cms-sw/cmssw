import FWCore.ParameterSet.Config as cms

process = cms.Process("TEST")
#process.load("MuonAnalysis.MomentumScaleCalibration.Summer08_Upsilon1S_cff")

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(

# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_10_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_11_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_12_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_13_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_14_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_15_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_16_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_17_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_18_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_19_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_1_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_20_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_21_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_22_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_23_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_24_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_25_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_26_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_27_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_28_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_2_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_3_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_4_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_5_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_6_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_7_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_8_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_FirstData2010_9_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_VeryFirstData2010_10_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_VeryFirstData2010_11_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_VeryFirstData2010_12_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_VeryFirstData2010_13_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_VeryFirstData2010_14_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_VeryFirstData2010_15_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_VeryFirstData2010_1_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_VeryFirstData2010_2_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_VeryFirstData2010_3_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_VeryFirstData2010_4_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_VeryFirstData2010_5_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_VeryFirstData2010_6_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_VeryFirstData2010_7_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_VeryFirstData2010_8_1.root",
# "rfio:/castor/cern.ch/user/c/covarell/temp/PAT_VeryFirstData2010_9_1.root",

"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV8_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v8_10_1.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV8_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v8_11_1.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV8_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v8_12_1.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV8_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v8_13_1.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV8_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v8_1_1.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV8_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v8_2_1.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV8_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v8_3_1.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV8_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v8_4_1.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV8_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v8_5_1.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV8_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v8_6_1.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV8_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v8_7_1.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV8_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v8_8_1.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV8_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v8_9_1.root",


"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_11_0.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_12_0.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_13_0.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_14_0.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_15_0.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_16_0.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_17_0.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_18_1.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_19_0.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_1_1.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_20_0.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_21_0.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_22_0.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_23_0.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_24_0.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_25_0.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_26_0.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_27_0.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_28_2.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_29_0.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_2_2.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_3_0.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_4_0.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_5_1.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_7_1.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_8_0.root",
"rfio:/castor/cern.ch/user/s/sbologne/cmst3/Data/CSOniaV9_lessCuts/PAT-MinimumBias-Commissioning10-CS_Onia-v9_9_0.root"


#NOT YET ALL THE STATISTICS for ppMuX
# "rfio:/castor/cern.ch/user/s/sbologne/cmst3/MCSpring10/ppMuX/PAT-ppMuX-Spring10-START3X_V26_S09-v1_11_1.root",
# "rfio:/castor/cern.ch/user/s/sbologne/cmst3/MCSpring10/ppMuX/PAT-ppMuX-Spring10-START3X_V26_S09-v1_12_1.root",
# "rfio:/castor/cern.ch/user/s/sbologne/cmst3/MCSpring10/ppMuX/PAT-ppMuX-Spring10-START3X_V26_S09-v1_14_1.root",
# "rfio:/castor/cern.ch/user/s/sbologne/cmst3/MCSpring10/ppMuX/PAT-ppMuX-Spring10-START3X_V26_S09-v1_15_1.root",
# "rfio:/castor/cern.ch/user/s/sbologne/cmst3/MCSpring10/ppMuX/PAT-ppMuX-Spring10-START3X_V26_S09-v1_16_1.root",
# "rfio:/castor/cern.ch/user/s/sbologne/cmst3/MCSpring10/ppMuX/PAT-ppMuX-Spring10-START3X_V26_S09-v1_18_1.root",
# "rfio:/castor/cern.ch/user/s/sbologne/cmst3/MCSpring10/ppMuX/PAT-ppMuX-Spring10-START3X_V26_S09-v1_19_1.root",
# "rfio:/castor/cern.ch/user/s/sbologne/cmst3/MCSpring10/ppMuX/PAT-ppMuX-Spring10-START3X_V26_S09-v1_1_1.root",
# "rfio:/castor/cern.ch/user/s/sbologne/cmst3/MCSpring10/ppMuX/PAT-ppMuX-Spring10-START3X_V26_S09-v1_20_1.root",
# "rfio:/castor/cern.ch/user/s/sbologne/cmst3/MCSpring10/ppMuX/PAT-ppMuX-Spring10-START3X_V26_S09-v1_21_1.root",
# "rfio:/castor/cern.ch/user/s/sbologne/cmst3/MCSpring10/ppMuX/PAT-ppMuX-Spring10-START3X_V26_S09-v1_22_1.root",
# "rfio:/castor/cern.ch/user/s/sbologne/cmst3/MCSpring10/ppMuX/PAT-ppMuX-Spring10-START3X_V26_S09-v1_23_1.root",
# "rfio:/castor/cern.ch/user/s/sbologne/cmst3/MCSpring10/ppMuX/PAT-ppMuX-Spring10-START3X_V26_S09-v1_24_1.root",
# "rfio:/castor/cern.ch/user/s/sbologne/cmst3/MCSpring10/ppMuX/PAT-ppMuX-Spring10-START3X_V26_S09-v1_25_1.root",
# "rfio:/castor/cern.ch/user/s/sbologne/cmst3/MCSpring10/ppMuX/PAT-ppMuX-Spring10-START3X_V26_S09-v1_2_1.root",
# "rfio:/castor/cern.ch/user/s/sbologne/cmst3/MCSpring10/ppMuX/PAT-ppMuX-Spring10-START3X_V26_S09-v1_4_1.root",
# "rfio:/castor/cern.ch/user/s/sbologne/cmst3/MCSpring10/ppMuX/PAT-ppMuX-Spring10-START3X_V26_S09-v1_6_1.root",
# "rfio:/castor/cern.ch/user/s/sbologne/cmst3/MCSpring10/ppMuX/PAT-ppMuX-Spring10-START3X_V26_S09-v1_8_1.root",
# "rfio:/castor/cern.ch/user/s/sbologne/cmst3/MCSpring10/ppMuX/PAT-ppMuX-Spring10-START3X_V26_S09-v1_9_1.root"

    ),
      lumisToProcess = cms.untracked.VLuminosityBlockRange('132440:157-132440:401',
                                                         '132442:1-132442:133',
                                                         '132442:136-132442:271',
                                                         '132596:382-132596:453',
                                                         '132597:1-132597:48',
                                                         '132598:1-132598:188',
                                                         '132599:1-132599:538',
                                                         '132601:1-132601:207',
                                                         '132601:209-132601:259',
                                                         '132601:261-132601:1131',
                                                         '132602:1-132602:83',
                                                         '132605:1-132605:829',
                                                         '132605:831-132605:968',
                                                         '132606:1-132606:37',
                                                         '132656:1-132656:140',
                                                         '132658:1-132658:177',
                                                         '132659:1-132659:84',
                                                         '132661:1-132661:130',
                                                         '132662:1-132662:130',
                                                         '132662:132-132662:165',
                                                         '132716:220-132716:640',
                                                         '132959:1-132959:417',
                                                         '132960:1-132960:190',
                                                         '132961:1-132961:427',
                                                         '132965:1-132965:107',
                                                         '132968:1-132968:173',
                                                         '133029:101-133029:115',
                                                         '133029:129-133029:350',
                                                         '133031:1-133031:18',
                                                         '133034:131-133034:325',
                                                         '133035:1-133035:306',
                                                         '133036:1-133036:225',
                                                         '133046:1-133046:43',
                                                         '133046:45-133046:323',
                                                         '133082:1-133082:608',
                                                         '133158:65-133158:786',
                                                         '133321:1-133321:383',
                                                         '133446:105-133446:273',
                                                         '133448:1-133448:516',
                                                         '133450:1-133450:658',
                                                         '133474:1-133474:95',
                                                         '133474:157-133474:189',
                                                         '133483:94-133483:591',
                                                         '133483:652-133483:658',
                                                         '133874:166-133874:875',
                                                         '133875:1-133875:49',
                                                         '133877:1-133877:77',
                                                         '133877:82-133877:231',
                                                         '133877:236-133877:1997',
                                                         '133881:1-133881:562',
                                                         '133885:1-133885:728',
                                                         '133927:1-133927:57',
                                                         '133928:1-133928:645')
)

process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
process.load("Geometry.CommonDetUnit.globalTrackingGeometry_cfi")
process.load("RecoMuon.DetLayers.muonDetLayerGeometry_cfi")
process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
process.load("RecoMuon.TrackingTools.MuonServiceProxy_cff")
process.load("Alignment.CommonAlignmentProducer.GlobalPosition_Fake_cff")
process.load("MagneticField.Engine.uniformMagneticField_cfi")

process.maxEvents = cms.untracked.PSet(
    # This are the total background events from InclusivePPmuX (89150) +
    # the number of Upsilon1S events.
    # input = cms.untracked.int32(89355)

    input = cms.untracked.int32(-1)
)
process.looper = cms.Looper(
    "MuScleFit",
    process.MuonServiceProxy,

    # Choose the kind of muons you want to run on
    # -------------------------------------------

    # // all muons //
    MuonLabel = cms.InputTag("patMuons"),
    # Defines what type of muons to use:
    # -1 = onia guys selection
    # -2 = onia guys selection - only GG
    # -3 = onia guys selection - only GT
    # -4 = onia guys selection - only TT
    # Note that the above samples are independent and represent the composition of the inclusive sample
    # 1 = global muon
    # 2 = standalone muon
    # 3 = tracker muon
    # 4 = calo muon
    # 10 = innerTrack of global muon
    MuonType = cms.int32(-1),

    # // standalone muons //
    # MuonLabel = cms.InputTag("standAloneMuons:UpdatedAtVtx"),
    # MuonType = cms.int32(2),

    # // tracker tracks //
    # MuonLabel = cms.InputTag("generalTracks"), # ctfWithMaterialTracks
    # MuonType = cms.int32(3),

    # // global muons (these are still reco::Tracks) //
    # MuonLabel = cms.InputTag("muons"),
    # MuonType = cms.int32(3),

    # Output settings
    # ---------------
    OutputFileName = cms.untracked.string('MuScleFitData.root'),
    debug = cms.untracked.int32(10),

    # Likelihood settings
    # -------------------
    maxLoopNumber = cms.untracked.int32(3),
    # Select which fits to do in which loop (0 = do not, 1 = do)
    doResolFit =      cms.vint32(0, 1, 0),
    doScaleFit =      cms.vint32(0, 0, 0),
    doBackgroundFit = cms.vint32(1, 0, 0),
    doCrossSectionFit = cms.vint32(0, 0, 0),

    # Fit parameters and fix flags (1 = use par)
    # ==========================================

    # BiasType=0 means no bias to muon momenta
    # ----------------------------------------
    BiasType = cms.int32(0),
    parBias = cms.vdouble(),

    # BiasType = 1 means linear bias on the muons Pt
    # the two parameters are the constant and the Pt
    # coefficient in this order.
    # ----------------------------------------------
    # BiasType = cms.int32(1),
    # parBias = cms.vdouble(1.015, 0.001),

    # Sinusoidal in phi
    # -----------------
    # BiasType = 3 means sinusoidal bias on the muons Pt
    # the two parameters are defined by:
    # pt = (parScale[0] + parScale[1]*sin(phi))*pt; 
    # BiasType = cms.int32(3),
    # parBias = cms.vdouble(1.015, 0.025),

    # SmearType=0 means no smearing applied to muon momenta
    # -----------------------------------------------------
    SmearType = cms.int32(0),
    parSmear = cms.vdouble(),

    # ------------------------- #
    # Resolution fit parameters #
    # ------------------------- #
    #ResolFitType = cms.int32(8),
    #parResol = cms.vdouble(-0.003, 0.000205, 1.0, 0.445473,
    #                       0.00043, 0.0041, 0.0000028, 0.000077,
    #                       0.00011, 0.0018, -0.00000094, 0.000022),
    #parResolFix = cms.vint32(0, 0, 0, 0,
    #                         1, 1, 1, 1,
    #                         1, 1, 1, 1),
    #parResolOrder = cms.vint32(0, 0, 0, 0,
    #                           0, 0, 0, 0,
    #                           0, 0, 0, 0),

    # ------------------------------------------------- #
    # New resolution function derived for low Pt region #
    # ------------------------------------------------- #
    ResolFitType = cms.int32(14), #inner tracks in 31X
    parResol = cms.vdouble(0.007,0.015, -0.00077, 0.0063, 0.0018, 0.0164),
    parResolFix = cms.vint32(0, 0, 0,0, 0,0),
    parResolOrder = cms.vint32(0, 0, 0, 0, 0, 0),


    # -------------------- #
    # Scale fit parameters #
    # -------------------- #

    # Scale fit type=14: Pt offset and grade up to three, Eta terms up to the sixth grade
    # -----------------------------------------------------------------------------------
    ScaleFitType = cms.int32(18),
    parScaleOrder = cms.vint32(0, 0, 0, 0),
    parScaleFix =   cms.vint32(0, 0, 0, 0),
    #parScale = cms.vdouble(1.0, -0.003, -0.0004, 0, 0),
    parScale = cms.vdouble(1, 1, 1, 1),

    # ---------------------------- #
    # Cross section fit parameters #
    # ---------------------------- #
    # Note that the cross section fit works differently than the others, it
    # fits ratios of parameters. Fix and Order should not be used as is, they
    # are there mainly for compatibility.
    parCrossSectionOrder = cms.vint32(0, 0, 0, 0, 0, 0),
    parCrossSectionFix =   cms.vint32(0, 0, 0, 0, 0, 0),
    parCrossSection = cms.vdouble(1.233, 2.07, 6.33, 13.9, 2.169, 127.2),

    # ------------------------- #
    # Background fit parameters #
    # ------------------------- #

    # Window factors for: Z, Upsilons and (J/Psi,Psi2S) regions
    LeftWindowBorder = cms.vdouble(70., 8., 1.391495),
    RightWindowBorder = cms.vdouble(110., 12., 5.391495),

    # The parameter of BgrFitType=1 is the bgr fraction
    # -------------------------------------------------
    # BgrFitType = cms.int32(1),
    # parBgrFix = cms.vint32(0),
    # parBgr = cms.vdouble(0.001),
    # parBgrOrder = cms.vint32(0),

    # The two parameters of BgrFitType=2 are respectively:
    # bgr fraction, (negative of) bgr exp. slope, bgr constant
    # --------------------------------------------------------
    # The function types for resonances in a region must be the same
    BgrFitType = cms.vint32(2, 2, 2), # regions
    # These empty parameters should be used when there is no background
    parBgr = cms.vdouble(0., 0.,   0., 0.,   0., 0.,
                         0., 0.,   0., 0.,   0., 0.,   0.,0.,   0.,0.,   0.,0.),
    parBgrFix = cms.vint32(0, 0,   0, 0,   0, 0,
                           # The rest of the parameters is used for the resonance regions. They are automatically fixed in the code
                           # because they are never used to fit the background, but only after the rescaling.
                           1, 1,   1, 1,   1, 1,   1, 1,   1, 1,   1, 1),
    parBgrOrder = cms.vint32(0, 0,   0, 0,   0, 0,
                             0, 0,   0, 0,   0, 0,   0, 0,   0, 0,   0, 0),

    # ---------------- #
    # Select resonance #
    # ---------------- #

    # The resonances are to be specified in this order:
    # Z0, Y(3S), Y(2S), Y(1S), Psi(2S), J/Psi
    # -------------------------------------------------
    resfind = cms.vint32(0, 0, 0, 0, 0, 1),
    FitStrategy = cms.int32(2),

    speedup = cms.bool(True),
    OutputGenInfoFileName = cms.untracked.string("genSimRecoPlotsData.root"),
    # Set this to false if you do not want to use simTracks.
    # (Note that this is skipped anyway if speedup == True).
    compareToSimTracks = cms.bool(True),

    # This line is only necessary when running on fastSim
    # SimTracksCollection = cms.untracked.InputTag("famosSimHits"),
    # This must be set to true when using events generated with Sherpa
    # Sherpa = cms.untracked.bool(True),

    # This line allows to switch to PAT muons. Default is false.
    PATmuons = cms.untracked.bool(True),

    # This line allows to use the EDLooper or to loop by hand.
    # All the necessary information is saved during the first loop so there is not need
    # at this time to read again the events in successive iterations. Therefore by default
    # for iterations > 1 the loops are done by hand, which means that the framework does
    # not need to read all the events again. This is much faster.
    # If you need to read the events in every iteration put this to false.
    # FastLoop = cms.untracked.bool(False),

    ProbabilitiesFileInPath = cms.untracked.string("MuonAnalysis/MomentumScaleCalibration/test/Probs_JPsiFSR.root"),

    # Only used when reading events from a root tree
    MaxEventsFromRootTree = cms.int32(-1),

    # Specify a file if you want to read events from a root tree in a local file.
    # In this case the input source should be an empty source with 0 events.
    InputRootTreeFileName = cms.string(""),
    # Specify the file name where you want to save a root tree with the muon pairs.
    # Leave empty if no file should be written.
    OutputRootTreeFileName = cms.string(""),

    # Fit accuracy and debug parameters
    StartWithSimplex = cms.bool(True),
    ComputeMinosErrors = cms.bool(False),
    MinimumShapePlots = cms.bool(False),

    # The following parameters can be used to filter events
    TriggerResultsLabel = cms.untracked.string("TriggerResults"),
    TriggerResultsProcess = cms.untracked.string("HLT"),
    # TriggerPath: "" = No trigger requirements, "All" = No specific path
    #TriggerPath = cms.untracked.string("HLT_L1MuOpen"),
    TriggerPath = cms.untracked.string("All"),
    # Negate the result of the trigger
    NegateTrigger = cms.untracked.bool(False),

)

# Timing information
#process.load("FWCore.MessageLogger.MessageLogger_cfi")
#process.MessageLogger.logMuScleFit = cms.PSet(
process.MessageLogger = cms.Service(
    "MessageLogger",
    destinations = cms.untracked.vstring('logMuScleFit'),
    # logMuScleFit = cms.untracked.PSet(
    threshold = cms.untracked.string('INFO'),
    #default = cms.untracked.PSet(

    #    limit = cms.untracked.int32(10000000)
    #    )
    # )
)

# TimingLogFile = cms.untracked.string('timing.log')
# process.Timing = cms.Service("Timing")
