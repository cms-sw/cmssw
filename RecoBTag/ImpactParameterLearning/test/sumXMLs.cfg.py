import FWCore.ParameterSet.Config as cms

process = cms.Process("sumXMLs")
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.source = cms.Source("EmptySource")

process.maxEvents = cms.untracked.PSet(
     input = cms.untracked.int32(1)
)
process.sumCalib = cms.EDAnalyzer("SumHistoCalibration",
	
	 xmlfiles2d = cms.vstring(
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-120to170_HLTFilter/2d_10_1_pB4.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-120to170_HLTFilter/2d_11_1_oka.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-120to170_HLTFilter/2d_12_1_9j0.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-120to170_HLTFilter/2d_13_1_wub.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-120to170_HLTFilter/2d_14_1_LjB.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-120to170_HLTFilter/2d_15_1_CpE.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_10_1_V1I.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_11_1_p6G.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_12_1_rvk.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_13_1_95Q.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_14_1_0JU.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_15_1_0ii.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_16_1_F6G.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_17_1_TUS.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_18_1_dnl.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_19_1_VbB.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_1_1_RuX.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_20_1_Oxb.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_21_1_ZkC.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_22_1_6T6.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_23_1_KUq.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_24_1_cFi.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_25_1_Mhq.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_26_1_W3a.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_27_1_qJQ.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_28_1_mVY.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_29_1_UAf.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_2_1_w8A.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_30_1_27k.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_31_1_vSH.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_32_1_h3Z.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_33_1_PFe.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_34_1_wfE.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_35_1_HlR.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_36_1_heS.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_37_1_LGX.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_38_1_0Pr.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_39_1_H9H.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_3_1_ySh.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_40_1_jG5.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_41_1_JlQ.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_42_1_ae0.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_43_1_ZDu.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_44_1_rIB.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_45_1_Eji.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_46_1_86o.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/2d_48_1_Qbz.xml"





	 ),
	 xmlfiles3d = cms.vstring(
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-120to170_HLTFilter/3d_10_1_pB4.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-120to170_HLTFilter/3d_11_1_oka.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-120to170_HLTFilter/3d_12_1_9j0.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-120to170_HLTFilter/3d_13_1_wub.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-120to170_HLTFilter/3d_14_1_LjB.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-120to170_HLTFilter/3d_15_1_CpE.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_10_1_V1I.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_11_1_p6G.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_12_1_rvk.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_13_1_95Q.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_14_1_0JU.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_15_1_0ii.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_16_1_F6G.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_17_1_TUS.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_18_1_dnl.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_19_1_VbB.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_1_1_RuX.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_20_1_Oxb.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_21_1_ZkC.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_22_1_6T6.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_23_1_KUq.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_24_1_cFi.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_25_1_Mhq.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_26_1_W3a.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_27_1_qJQ.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_28_1_mVY.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_29_1_UAf.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_2_1_w8A.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_30_1_27k.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_31_1_vSH.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_32_1_h3Z.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_33_1_PFe.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_34_1_wfE.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_35_1_HlR.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_36_1_heS.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_37_1_LGX.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_38_1_0Pr.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_39_1_H9H.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_3_1_ySh.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_40_1_jG5.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_41_1_JlQ.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_42_1_ae0.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_43_1_ZDu.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_44_1_rIB.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_45_1_Eji.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_46_1_86o.xml",
	 "RecoBTag/ImpactParameterLearning/test/QCD_Pt-80to120_HLTFilter/3d_48_1_Qbz.xml"

	 ),
	 sum2D = cms.bool(True),
	 sum3D = cms.bool(True),
         writeToDB       = cms.bool(True),
         writeToRootXML  = cms.bool(False),
         writeToBinary   = cms.bool(False)
)



process.PoolDBOutputService = cms.Service("PoolDBOutputService",
    authenticationMethod = cms.untracked.uint32(1),
    loadBlobStreamer = cms.untracked.bool(True),
    catalog = cms.untracked.string('file:mycatalog_new.xml'),
    DBParameters = cms.PSet(
        messageLevel = cms.untracked.int32(0),
        authenticationPath = cms.untracked.string('.')
    ),
    timetype = cms.untracked.string('runnumber'),
    connect = cms.string('sqlite_file:btagnew_MC53X.db'),
    toPut = cms.VPSet(cms.PSet(
        record = cms.string('BTagTrackProbability2DRcd'),
        tag = cms.string('probBTagPDF2D_tag_mc')
    ), 
        cms.PSet(
            record = cms.string('BTagTrackProbability3DRcd'),
            tag = cms.string('probBTagPDF3D_tag_mc')
        ))
)

process.p = cms.Path(process.sumCalib)
