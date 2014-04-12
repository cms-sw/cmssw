# PYTHON configuration file for class: CMSDAS11DijetAnalyzer.cc
# Description:  Example of simple EDAnalyzer for dijet mass & dijet spectrum ratio analysis
# Authors: J.P. Chou, Jason St. John
# Date:  01 - January - 2011
import FWCore.ParameterSet.Config as cms

process = cms.Process("Ana")
process.load("FWCore.MessageService.MessageLogger_cfi")


# Which files to use
# 1: data, 10k events
# 2: MC resonance 1.2TeV
# 3: ??
whichfiles = 2;

if (whichfiles==1):
    thefileNames = cms.untracked.vstring('file:/uscms_data/d2/kalanand/dijet-Run2010A-JetMET-Nov4ReReco-9667events.root')
elif (whichfiles==2):
    thefileNames = cms.untracked.vstring('/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0026/6C0BC238-4247-DF11-81D7-E41F1318160C.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/FE3B276E-1C47-DF11-83B7-00215E22053A.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/FCD94068-2647-DF11-80FF-E41F131817F8.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/F843DD92-2647-DF11-9B8F-E41F13181668.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/F62E9060-1C47-DF11-B431-00215E221BC0.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/DA63FD75-1C47-DF11-A347-00215E221692.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/D802BE12-1D47-DF11-9FA1-00215E93D738.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/D65DA979-1C47-DF11-941C-00215E21DD56.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/D2647956-1C47-DF11-9C8E-00215E2216EC.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/D0A03150-1C47-DF11-ACE6-E41F1318099C.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/CC2D830B-1D47-DF11-966E-E41F13181498.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/CACB401B-1847-DF11-B9C5-E41F13181D00.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/CA85A868-2647-DF11-86BA-00215E221818.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/CA4A0A60-1C47-DF11-836E-00215E21D86A.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/CA070629-2747-DF11-819C-00215E222790.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/C0757277-1C47-DF11-90B7-00215E2212D2.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/BE19E200-1D47-DF11-A019-00215E22200A.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/AE023D13-1D47-DF11-892F-E41F13181688.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/AC98A079-1E47-DF11-8842-E41F13180A64.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/AA56B163-1C47-DF11-BF57-00215E21D570.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/9E440461-1C47-DF11-AC32-00215E21DBA0.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/9A160E70-1C47-DF11-AE68-00215E2208EE.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/96DF3203-1D47-DF11-849B-00215E21D948.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/9606A290-2647-DF11-A095-E41F13181AB4.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/9451B34F-1C47-DF11-B75C-E41F13181A5C.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/80CCEE69-1E47-DF11-9216-00215E2211F4.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/76B9DA6A-1C47-DF11-B13C-00215E222340.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/74489C67-2647-DF11-9B52-E41F13181588.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/6E8FD06A-1C47-DF11-A1DF-00215E21D9F6.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/6E872979-1C47-DF11-87DE-00215E21D540.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/646F4C0E-1D47-DF11-9C8E-00215E21D7C8.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/62D1695A-1C47-DF11-9C53-00215E21D786.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/5CAAC1A3-1B47-DF11-9B5B-00215E21DBBE.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/5A173FCA-2547-DF11-8F56-E41F131816A0.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/5670C812-1D47-DF11-8020-E41F13181890.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/52597D68-1E47-DF11-B7D1-E41F131816B4.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/4ECB6F7B-1E47-DF11-833D-E41F131816A0.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/4EB42E67-1C47-DF11-A1FA-00215E2205AC.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/4EB42E67-1C47-DF11-A1FA-00215E2205AC.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/4AEB1C64-1C47-DF11-BA6A-00215E221B48.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/48CCB914-1D47-DF11-B3D2-E41F13181044.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/3E489C57-1C47-DF11-A6EC-00215E93ED9C.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/3C04D1FE-1C47-DF11-8E76-E41F1318170C.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/3AE57490-2647-DF11-AA52-E41F13181CF8.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/38AC77A1-1B47-DF11-B015-E41F13181CA4.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/2AD2CC1E-1847-DF11-BCE3-00215E21D57C.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/2A4DFD09-1D47-DF11-9D16-00215E2223D6.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/2A407379-1C47-DF11-A7F2-00215E21DAF2.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/289FB808-1D47-DF11-A091-00215E222808.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/28160333-1847-DF11-BAD5-00215E21D702.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/220A4374-1C47-DF11-9FA1-00215E21DF18.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/20D4E62F-1847-DF11-8D41-00215E2222A4.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/1C8DD3A3-1B47-DF11-BAF8-E41F1318168C.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/169CCA06-1D47-DF11-B2C0-00215E221EEA.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/102D9F52-1C47-DF11-B437-00215E22181E.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/0EB6DB6E-1E47-DF11-8D0B-00215E93DCFC.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/08056A58-1C47-DF11-A42D-00215E21DAAA.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0025/06E9D555-1C47-DF11-9E40-00215E2219E6.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0024/E2CAFC76-0C47-DF11-9784-00215E93E7DC.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0024/94318864-1147-DF11-B2FD-E41F131815B8.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0024/66451B78-0C47-DF11-8CB0-00215E93EE44.root',
                                      '/store/mc/Spring10/Qstar_DiJet1200/GEN-SIM-RECO/START3X_V26_S09-v1/0024/5A3665E0-1047-DF11-9537-00215E22175E.root'
                                      )
    
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

#############   Set the number of events #############
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(10000)
)
#############   Define the source file ###############
process.source = cms.Source("PoolSource",
#                             fileNames = cms.untracked.vstring('/store/mc/Fall10/QCD_Pt_80to120_TuneZ2_7TeV_pythia6/GEN-SIM-RECO/START38_V12-v1/0000/FEF4D100-4CCB-DF11-94CB-00E08178C12F.root')
                            fileNames = thefileNames
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
                                  innerDeltaEta = cms.double(0.7),
                                  outerDeltaEta = cms.double(1.3),
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


