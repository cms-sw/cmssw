# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()


overridesEv5={'-n':'5'}

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

## production tests
workflows[1] = ['', ['ProdMinBias','DIGIPROD1','RECOPROD1']]
workflows[2] = ['', ['ProdTTbar','DIGIPROD1','RECOPROD1']]
workflows[3] = ['', ['ProdQCD_Pt_3000_3500','DIGIPROD1','RECOPROD1']]
workflows.addOverride(3,overridesEv5)
workflows[1301] = ['', ['ProdMinBias_13','DIGIUP15PROD1','RECOPRODUP15']]
workflows[1302] = ['', ['ProdTTbar_13','DIGIUP15PROD1','RECOPRODUP15']]
workflows[1303] = ['', ['ProdQCD_Pt_3000_3500_13','DIGIUP15PROD1','RECOPRODUP15']]
workflows.addOverride(1303,overridesEv5)

### data ###
workflows[4.5]  = ['', ['RunCosmicsA','RECOCOSD','ALCACOSD','HARVESTDC']]
workflows[4.6]  = ['', ['MinimumBias2010A','RECOSKIM','HARVESTD']]
#workflows[4.7]  = ['', ['MinimumBias2010B','RECOSKIMALCA','HARVESTD']]
#workflows[4.8]  = ['', ['WZMuSkim2010A','RECOSKIM','HARVESTD']]
#workflows[4.9]  = ['', ['WZEGSkim2010A','RECOSKIM','HARVESTD']]
#workflows[4.10] = ['', ['WZMuSkim2010B','RECOSKIM','HARVESTD']]
#workflows[4.11] = ['', ['WZEGSkim2010B','RECOSKIM','HARVESTD']]

workflows[4.12] = ['', ['RunMinBias2010B','RECOD','HARVESTD']]
#workflows[4.13] = ['', ['RunMu2010B','RECOD','HARVESTD']]
#workflows[4.14] = ['', ['RunElectron2010B','RECOD','HARVESTD']]
#workflows[4.15] = ['', ['RunPhoton2010B','RECOD','HARVESTD']]
#workflows[4.16] = ['', ['RunJet2010B','RECOD','HARVESTD']]


workflows[4.17] = ['', ['RunMinBias2011A','HLTD','RECODreHLT','HARVESTDreHLT','SKIMDreHLT']]
workflows[4.18] = ['', ['RunMu2011A','RECOD','HARVESTD']]
workflows[4.19] = ['', ['RunElectron2011A','RECOD','HARVESTD']]
workflows[4.20] = ['', ['RunPhoton2011A','RECOD','HARVESTD']]
workflows[4.21] = ['', ['RunJet2011A','RECOD','HARVESTD']]

workflows[4.22] = ['', ['RunCosmics2011A','RECOCOSD','ALCACOSD','SKIMCOSD','HARVESTDC']]
workflows[4.23] = ['',['ValSkim2011A','HLTDSKIM','RECODreHLT','HARVESTDreHLT']]
workflows[4.24] = ['',['WMuSkim2011A','HLTDSKIM','RECODreHLT','HARVESTDreHLT']]
workflows[4.25] = ['',['WElSkim2011A','HLTDSKIM','RECODreHLT','HARVESTDreHLT']]
workflows[4.26] = ['',['ZMuSkim2011A','HLTDSKIM','RECODreHLT','HARVESTDreHLT']]
workflows[4.27] = ['',['ZElSkim2011A','HLTDSKIM','RECODreHLT','HARVESTDreHLT']]
workflows[4.28] = ['',['HighMet2011A','HLTDSKIM','RECODreHLT','HARVESTDreHLT']]

workflows[4.29] = ['', ['RunMinBias2011B','HLTD','RECODreHLT','HARVESTDreHLT','SKIMDreHLT']]
#workflows[4.291] = ['', ['RunMinBias2011B','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.30] = ['', ['RunMu2011B','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.31] = ['', ['RunElectron2011B','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.32] = ['', ['RunPhoton2011B','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.33] = ['', ['RunJet2011B','HLTD','RECODreHLT','HARVESTDreHLT']]

workflows[4.34] = ['',['ValSkim2011B','HLTDSKIM','RECODreHLT','HARVESTDreHLT']]
workflows[4.35] = ['',['WMuSkim2011B','HLTDSKIM','RECODreHLT','HARVESTDreHLT']]
workflows[4.36] = ['',['WElSkim2011B','HLTDSKIM','RECODreHLT','HARVESTDreHLT']]
workflows[4.37] = ['',['ZMuSkim2011B','HLTDSKIM','RECODreHLT','HARVESTDreHLT']]
workflows[4.38] = ['',['ZElSkim2011B','HLTDSKIM','RECODreHLT','HARVESTDreHLT']]
workflows[4.39] = ['',['HighMet2011B','HLTDSKIM','RECODreHLT','HARVESTDreHLT']]

workflows[4.40] = ['',['RunMinBias2012A','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.41] = ['',['RunTau2012A','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.42] = ['',['RunMET2012A','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.43] = ['',['RunMu2012A','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.44] = ['',['RunElectron2012A','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.45] = ['',['RunJet2012A','HLTD','RECODreHLT','HARVESTDreHLT']]

workflows[4.51] = ['',['RunMinBias2012B','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.52] = ['',['RunMu2012B','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.53] = ['',['RunPhoton2012B','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.54] = ['',['RunEl2012B','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.55] = ['',['RunJet2012B','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.56] = ['',['ZMuSkim2012B','HLTDSKIM','RECODreHLT','HARVESTDreHLT']]
workflows[4.57] = ['',['ZElSkim2012B','HLTDSKIM','RECODreHLT','HARVESTDreHLT']]
workflows[4.58] = ['',['WElSkim2012B','HLTDSKIM','RECODreHLT','HARVESTDreHLT']]

workflows[4.61] = ['',['RunMinBias2012C','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.62] = ['',['RunMu2012C','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.63] = ['',['RunPhoton2012C','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.64] = ['',['RunEl2012C','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.65] = ['',['RunJet2012C','HLTD','RECODreHLT','HARVESTDreHLT']]

workflows[4.66] = ['',['ZMuSkim2012C','HLTDSKIM','RECODreHLT','HARVESTDreHLT']]
workflows[4.67] = ['',['ZElSkim2012C','HLTDSKIM','RECODreHLT','HARVESTDreHLT']]
workflows[4.68] = ['',['WElSkim2012C','HLTDSKIM','RECODreHLT','HARVESTDreHLT']]

workflows[4.71] = ['',['RunMinBias2012D','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.72] = ['',['RunMu2012D','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.73] = ['',['RunPhoton2012D','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.74] = ['',['RunEl2012D','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.75] = ['',['RunJet2012D','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.76] = ['',['ZMuSkim2012D','HLTDSKIM','RECODreHLT','HARVESTDreHLT']]
workflows[4.77] = ['',['ZElSkim2012D','HLTDSKIM','RECODreHLT','HARVESTDreHLT']]
workflows[4.78] = ['',['WElSkim2012D','HLTDSKIM','RECODreHLT','HARVESTDreHLT']]

workflows[140.51] = ['',['RunHI2010','REPACKHID','RECOHID11St3','HARVESTDHI']]
workflows[140.52] = ['',['RunHI2010','RECOHID10','RECOHIR10D11','HARVESTDHI']]
workflows[140.53] = ['',['RunHI2011','RECOHID11','HARVESTDHI']]

### run2 2015b 50ns ###
workflows[134.701] = ['',['RunHLTPhy2015B','HLTDR250ns','RECODR250nsreHLT','HARVESTDR2reHLT']]
workflows[134.702] = ['',['RunDoubleEG2015B','HLTDR250ns','RECODR250nsreHLT','HARVESTDR2reHLT']]
workflows[134.703] = ['',['RunDoubleMuon2015B','HLTDR250ns','RECODR250nsreHLT','HARVESTDR2reHLT']]
workflows[134.704] = ['',['RunJetHT2015B','HLTDR250ns','RECODR250nsreHLT','HARVESTDR2reHLT']]
workflows[134.705] = ['',['RunMET2015B','HLTDR250ns','RECODR250nsreHLT','HARVESTDR2reHLT']]
workflows[134.706] = ['',['RunMuonEG2015B','HLTDR250ns','RECODR250nsreHLT','HARVESTDR2reHLT']]

### run 2015C 25ns ###
workflows[134.801] = ['',['RunHLTPhy2015C','HLTDR225ns','RECODR225nsreHLT','HARVESTDR2reHLT']]
workflows[134.802] = ['',['RunDoubleEG2015C','HLTDR225ns','RECODR225nsreHLT','HARVESTDR2reHLT']]
workflows[134.803] = ['',['RunDoubleMuon2015C','HLTDR225ns','RECODR225nsreHLT','HARVESTDR2reHLT']]
workflows[134.804] = ['',['RunJetHT2015C','HLTDR225ns','RECODR225nsreHLT','HARVESTDR2reHLT']]
workflows[134.805] = ['',['RunMET2015C','HLTDR225ns','RECODR225nsreHLT','HARVESTDR2reHLT']]
workflows[134.806] = ['',['RunMuonEG2015C','HLTDR225ns','RECODR225nsreHLT','HARVESTDR2reHLT']]
workflows[134.807] = ['',['RunDoubleEGPrpt2015C','HLTDR225ns','RECODR225nsreHLT','HARVESTDR2reHLT']]
workflows[134.808] = ['',['RunSingleMuPrpt2015C','HLTDR225ns','RECODR225nsreHLT','HARVESTDR2reHLT']]


### fastsim ###
workflows[5.1] = ['TTbar', ['TTbarFS','HARVESTFS']]
workflows[5.2] = ['SingleMuPt10', ['SingleMuPt10FS','HARVESTFS']]
workflows[5.3] = ['SingleMuPt100', ['SingleMuPt100FS','HARVESTFS']]
workflows[5.4] = ['ZEE', ['ZEEFS','HARVESTFS']]
workflows[5.5] = ['ZTT',['ZTTFS','HARVESTFS']]

workflows[5.6]  = ['QCD_FlatPt_15_3000', ['QCDFlatPt153000FS','HARVESTFS']]
workflows[5.7] = ['H130GGgluonfusion', ['H130GGgluonfusionFS','HARVESTFS']]

### fastsim_13 TeV ###
workflows[135.1] = ['TTbar_13', ['TTbarFS_13','HARVESTUP15FS','MINIAODMCUP15FS']]
workflows[135.2] = ['SingleMuPt10_UP15', ['SingleMuPt10FS_UP15','HARVESTUP15FS','MINIAODMCUP15FS']]
workflows[135.3] = ['SingleMuPt100_UP15', ['SingleMuPt100FS_UP15','HARVESTUP15FS','MINIAODMCUP15FS']]
workflows[135.4] = ['ZEE_13', ['ZEEFS_13','HARVESTUP15FS','MINIAODMCUP15FS']]
workflows[135.5] = ['ZTT_13',['ZTTFS_13','HARVESTUP15FS','MINIAODMCUP15FS']]
workflows[135.9] = ['ZMM_13',['ZMMFS_13','HARVESTUP15FS','MINIAODMCUP15FS']]

workflows[135.6]  = ['QCD_FlatPt_15_3000_13', ['QCDFlatPt153000FS_13','HARVESTUP15FS','MINIAODMCUP15FS']]
workflows[135.7] = ['H130GGgluonfusion_13', ['H130GGgluonfusionFS_13','HARVESTUP15FS','MINIAODMCUP15FS']]
workflows[135.11] = ['SMS-T1tttt_mGl-1500_mLSP-100_13', ['SMS-T1tttt_mGl-1500_mLSP-100FS_13','HARVESTUP15FS','MINIAODMCUP15FS']]


### MinBias fastsim_13 TeV for mixing ###
workflows[135.8] = ['',['MinBiasFS_13_ForMixing']]

### standard set ###
## particle guns
workflows[15] = ['', ['SingleElectronPt10','DIGI','RECO','HARVEST']]
workflows[16] = ['', ['SingleElectronPt1000','DIGI','RECO','HARVEST']]
workflows[17] = ['', ['SingleElectronPt35','DIGI','RECO','HARVEST']]
workflows[18] = ['', ['SingleGammaPt10','DIGI','RECO','HARVEST']]
workflows[19] = ['', ['SingleGammaPt35','DIGI','RECO','HARVEST']]
workflows[6]  = ['', ['SingleMuPt1','DIGI','RECO','HARVEST']]
workflows[20] = ['', ['SingleMuPt10','DIGI','RECO','HARVEST']]
workflows[21] = ['', ['SingleMuPt100','DIGI','RECO','HARVEST']]
workflows[22] = ['', ['SingleMuPt1000','DIGI','RECO','HARVEST']]
## particle guns postLS1
workflows[1315] = ['', ['SingleElectronPt10_UP15','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1316] = ['', ['SingleElectronPt1000_UP15','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1317] = ['', ['SingleElectronPt35_UP15','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1318] = ['', ['SingleGammaPt10_UP15','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1319] = ['', ['SingleGammaPt35_UP15','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1306]  = ['', ['SingleMuPt1_UP15','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1320] = ['', ['SingleMuPt10_UP15','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1321] = ['', ['SingleMuPt100_UP15','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1322] = ['', ['SingleMuPt1000_UP15','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1323] = ['', ['NuGun_UP15','DIGIUP15','RECOUP15','HARVESTUP15']]

## 8 TeV
workflows[24] = ['', ['TTbarLepton','DIGI','RECO','HARVEST']]
workflows[35] = ['', ['Wjet_Pt_80_120','DIGI','RECO','HARVEST']]
workflows[36] = ['', ['Wjet_Pt_3000_3500','DIGI','RECO','HARVEST']]
workflows[37] = ['', ['LM1_sfts','DIGI','RECO','HARVEST']]
# the input for the following worrkflow is high statistics
workflows[38] = ['', ['QCD_FlatPt_15_3000HS','DIGI','RECO','HARVEST']]

workflows[9]  = ['', ['Higgs200ChargedTaus','DIGI','RECO','HARVEST']]
workflows[13] = ['', ['QCD_Pt_3000_3500','DIGI','RECO','HARVEST']]
workflows.addOverride(13,overridesEv5)
workflows[39] = ['', ['QCD_Pt_600_800','DIGI','RECO','HARVEST']]
workflows[23] = ['', ['JpsiMM','DIGI','RECO','HARVEST']]
workflows[25] = ['', ['TTbar','DIGI','RECOAlCaEle','HARVEST','ALCATT']]
workflows[26] = ['', ['WE','DIGI','RECOAlCaEle','HARVEST']]
workflows[29] = ['', ['ZEE','DIGI','RECOAlCaEle','HARVEST']]
workflows[31] = ['', ['ZTT','DIGI','RECO','HARVEST']]
workflows[32] = ['', ['H130GGgluonfusion','DIGI','RECO','HARVEST']]
workflows[33] = ['', ['PhotonJets_Pt_10','DIGI','RECO','HARVEST']]
workflows[34] = ['', ['QQH1352T','DIGI','RECO','HARVEST']]
#workflows[46] = ['', ['ZmumuJets_Pt_20_300']]

workflows[7]  = ['', ['Cosmics','DIGICOS','RECOCOS','ALCACOS','HARVESTCOS']]
workflows[8]  = ['', ['BeamHalo','DIGICOS','RECOCOS','ALCABH','HARVESTCOS']]
workflows[11] = ['', ['MinBias','DIGI','RECOMIN','HARVEST','ALCAMIN']]
workflows[28] = ['', ['QCD_Pt_80_120','DIGI','RECO','HARVEST']]
workflows[27] = ['', ['WM','DIGI','RECO','HARVEST']]
workflows[30] = ['', ['ZMM','DIGI','RECO','HARVEST']]

workflows[10] = ['', ['ADDMonoJet_d3MD3','DIGI','RECO','HARVEST']]
workflows[12] = ['', ['ZpMM','DIGI','RECO','HARVEST']]
workflows[14] = ['', ['WpM','DIGI','RECO','HARVEST']]

workflows[43] = ['', ['ZpMM_2250_8TeV','DIGI','RECO','HARVEST']]
workflows[44] = ['', ['ZpEE_2250_8TeV','DIGI','RECO','HARVEST']]
workflows[45] = ['', ['ZpTT_1500_8TeV','DIGI','RECO','HARVEST']]

## 13 TeV and postLS1 geometry
workflows[1324] = ['', ['TTbarLepton_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1335] = ['', ['Wjet_Pt_80_120_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1336] = ['', ['Wjet_Pt_3000_3500_13','DIGIUP15','RECOUP15','HARVESTUP15']]
#workflows[1337] = ['', ['LM1_sfts_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1337] = ['', ['SMS-T1tttt_mGl-1500_mLSP-100_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1338] = ['', ['QCD_FlatPt_15_3000HS_13','DIGIUP15','RECOUP15','HARVESTUP15']]

workflows[1309]  = ['', ['Higgs200ChargedTaus_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1313] = ['', ['QCD_Pt_3000_3500_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows.addOverride(1313,overridesEv5)
workflows[1339] = ['', ['QCD_Pt_600_800_13','DIGIUP15','RECOUP15','HARVESTUP15']]

workflows[1347] = ['', ['Upsilon1SToMuMu_13','DIGIUP15','RECOUP15','HARVESTUP15']]
#workflows[1348] = ['', ['BuToKstarMuMu_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1348] = ['', ['EtaBToJpsiJpsi_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1349] = ['', ['BsToMuMu_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1350] = ['', ['JpsiMuMu_Pt-15','DIGIUP15','RECOUP15','HARVESTUP15']]
#workflows[1351] = ['', ['BuToKstarPsi2S_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1351] = ['', ['BuMixing_13','DIGIUP15','RECOUP15','HARVESTUP15']]

workflows[1325] = ['', ['TTbar_13','DIGIUP15','RECOUP15','HARVESTUP15','ALCATTUP15']]
workflows[1326] = ['', ['WE_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1329] = ['', ['ZEE_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1331] = ['', ['ZTT_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1332] = ['', ['H130GGgluonfusion_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1333] = ['', ['PhotonJets_Pt_10_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1334] = ['', ['QQH1352T_13','DIGIUP15','RECOUP15','HARVESTUP15']]

#workflows[1307]  = ['', ['Cosmics_UP15','DIGIHAL','RECOHAL','ALCAHAL','HARVESTHAL']]
workflows[1307]= ['', ['CosmicsSPLoose_UP15','DIGIHAL','RECOHAL','ALCAHAL','HARVESTHAL']]
workflows[1308]  = ['', ['BeamHalo_13','DIGIHAL','RECOHAL','ALCAHAL','HARVESTHAL']]
workflows[1311] = ['', ['MinBias_13','DIGIUP15','RECOMINUP15','HARVESTUP15','ALCAMIN']]
workflows[1328] = ['', ['QCD_Pt_80_120_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1327] = ['', ['WM_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1330] = ['', ['ZMM_13','DIGIUP15','RECOUP15','HARVESTUP15']]

workflows[1310] = ['', ['ADDMonoJet_d3MD3_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1312] = ['', ['ZpMM_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1314] = ['', ['WpM_13','DIGIUP15','RECOUP15','HARVESTUP15']]

#workflows[1340] = ['', ['BuJpsiK_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1340] = ['', ['PhiToMuMu_13','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1341] = ['', ['RSKKGluon_m3000GeV_13','DIGIUP15','RECOUP15','HARVESTUP15']]

workflows[1343] = ['', ['ZpMM_2250_13TeV','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1344] = ['', ['ZpEE_2250_13TeV','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1345] = ['', ['ZpTT_1500_13TeV','DIGIUP15','RECOUP15','HARVESTUP15']]

workflows[1352] = ['HSCPstop_M_200_13TeV', ['HSCPstop_M_200_13TeV','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1353] = ['RSGravitonToGaGa_13TeV', ['RSGravitonToGaGa_13TeV','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[1354] = ['WpToENu_M-2000_13TeV', ['WpToENu_M-2000_13TeV','DIGIUP15','RECOUP15','HARVESTUP15']]

### HI test ###
workflows[140] = ['',['HydjetQ_MinBias_2760GeV','DIGIHI','RECOHI','HARVESTHI']]
workflows[140.1] = ['',['QCD_Pt_80_120_13_HI','DIGIHI','RECOHI','HARVESTHI']]
workflows[140.2] = ['',['PhotonJets_Pt_10_13_HI','DIGIHI','RECOHI','HARVESTHI']]
workflows[140.3] = ['',['ZMM_13_HI','DIGIHI','RECOHI','HARVESTHI']]
workflows[140.4] = ['',['ZEEMM_13_HI','DIGIHI','RECOHI','HARVESTHI']]

# legacy: 141 => B0; 143 => B3 ; 144 => B5
#workflows[142] = ['',['HydjetQ_B8_2760GeV','DIGIHI','RECOHI','HARVESTHI','MINIAODMCUP15HI']]

### pPb test ###
workflows[280]= ['',['AMPT_PPb_5020GeV_MinimumBias','DIGI','RECO','HARVEST']]
