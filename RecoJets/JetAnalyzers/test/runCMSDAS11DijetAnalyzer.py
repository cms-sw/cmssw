# PYTHON configuration file for class: CMSDAS11DijetAnalyzer.cc
# Description:  Example of simple EDAnalyzer for dijet mass & dijet spectrum ratio analysis
# Authors: J.P. Chou, Jason St. John
# Date:  01 - January - 2011
import FWCore.ParameterSet.Config as cms

process = cms.Process("Ana")
process.load("FWCore.MessageService.MessageLogger_cfi")
#############   Set the number of events #############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
#############   Define the source file ###############
process.source = cms.Source("PoolSource",
#    fileNames = cms.untracked.vstring('/store/mc/Fall10/QCD_Pt_80to120_TuneZ2_7TeV_pythia6/GEN-SIM-RECO/START38_V12-v1/0000/FEF4D100-4CCB-DF11-94CB-00E08178C12F.root')
     fileNames = cms.untracked.vstring('file:/uscms_data/d2/kalanand/dijet-Run2010A-JetMET-Nov4ReReco-9667events.root')
#     fileNames = cms.untracked.vstring('file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_10_1_91x.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_11_1_wXT.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_12_1_MeU.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_13_1_lCw.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_14_1_NmK.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_15_1_aqj.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_16_1_WzF.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_17_1_0Gh.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_18_1_Fch.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_19_1_u0m.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_1_1_I0F.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_20_1_wom.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_21_1_CLq.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_22_1_rCM.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_23_1_6q6.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_24_1_Qq5.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_25_1_miT.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_26_1_OAB.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_27_1_yTU.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_28_1_Dy7.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_29_1_l4l.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_2_1_BNj.root', 
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_30_1_eY4.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_31_1_qgM.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_32_1_rmg.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_33_1_XvI.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_34_1_w34.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_35_1_VHA.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_36_1_y8O.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_37_1_Ohb.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_38_1_FyD.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_39_1_9Pd.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_3_1_CHK.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_40_1_Ee2.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_41_1_o0D.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_42_1_3NF.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_43_1_brY.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_44_1_us7.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_45_1_ILW.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_46_1_8aF.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_47_1_qM4.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_48_1_hRB.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_49_1_IHe.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_4_1_NDL.root', 
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_50_1_jXW.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_51_1_blV.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_52_1_guT.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_53_1_U0e.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_54_1_UuA.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_55_1_ivP.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_56_1_cbL.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_57_1_oDp.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_58_1_ie6.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_59_1_WuS.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_5_1_l5k.root', 
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_60_1_HKs.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_61_1_I4n.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_62_1_QRH.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_63_1_wKO.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_6_1_5wh.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_7_1_pdK.root', 
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_8_1_Tuu.root',
#                                       'file:/pnfs/cms/WAX/resilient/kalanand/JetSkim/JetSkim_9_1_Dcz.root'
#                                       )                          
)                                      

##-------------------- Communicate with the DB -----------------------
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
process.GlobalTag.globaltag = 'START38_V13::All'

#############   Include the jet corrections ##########
process.load("JetMETCorrections.Configuration.DefaultJEC_cff")
# set the record's IOV. Must be defined once. Choose ANY correction service. #

#############   Correct Calo Jets on the fly #########
process.dijetAna = cms.EDAnalyzer("CMSDAS11DijetAnalyzer",
                                  jetSrc = cms.InputTag("ak7CaloJets"),
                                  vertexSrc = cms.InputTag("offlinePrimaryVertices"),
                                  jetCorrections = cms.string("ak7CaloL2L3Residual"),
                                  innerDeltaEta = cms.double(1.3),
                                  outerDeltaEta = cms.double(3.0),
                                  JESbias = cms.double(1.0)
)

#############   Path       ###########################
process.p = cms.Path(process.dijetAna)


#############   Format MessageLogger #################
process.MessageLogger.cerr.FwkReport.reportEvery = 10

#############  This is how CMS handles output ROOT files #################
process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string("histos.root")
 )


