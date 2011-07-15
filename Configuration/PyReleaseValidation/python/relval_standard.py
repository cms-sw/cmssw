
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = {}

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

## production tests
workflows[1] = ['', ['ProdMinBias','DIGIPROD1','RECOPROD1']]
workflows[2] = ['', ['ProdTTbar','DIGIPROD1','RECOPROD1']]
workflows[3] = ['', ['ProdQCD_Pt_3000_3500','DIGIPROD1','RECOPROD1']]

### data ###
workflows[4.5]  = ['', ['RunCosmicsA','RECOCOSD','ALCACOSD']]
workflows[4.45] = ['', ['RunCosmicsA','RECOD']]
workflows[4.6]  = ['', ['MinimumBias2010A','RECOVALSKIM']]
workflows[4.7]  = ['', ['MinimumBias2010B','RECOVALSKIMALCA']]
workflows[4.8]  = ['', ['WZMuSkim2010A','RECOVALSKIM']]
workflows[4.9]  = ['', ['WZEGSkim2010A','RECOVALSKIM']]
workflows[4.10] = ['', ['WZMuSkim2010B','RECOVALSKIM']]
workflows[4.11] = ['', ['WZEGSkim2010B','RECOVALSKIM']]
workflows[4.12] = ['', ['RunMinBias2010B','RECOD']]
workflows[4.13] = ['', ['RunMu2010B','RECOD']]
workflows[4.14] = ['', ['RunElectron2010B','RECOD']]
workflows[4.15] = ['', ['RunPhoton2010B','RECOD']]
workflows[4.16] = ['', ['RunJet2010B','RECOD']]

### fastsim ###
workflows[5.1] = ['TTbar', ['TTbarFS1']]
workflows[6.3] = ['TTbar', ['TTbarFS2']]
workflows[5.2] = ['SingleMuPt10', ['SingleMuPt10FS']]
workflows[5.3] = ['SingleMuPt100', ['SingleMuPt100FS']]
workflows[6.1] = ['ZEE', ['ZEEFS1']]
workflows[6.2] = ['ZEE', ['ZEEFS2']]
workflows[39]  = ['QCDFlatPt153000', ['QCDFlatPt153000FS']]
workflows[6.4] = ['H130GGgluonfusion', ['H130GGgluonfusionFS']]

### standard set ###
workflows[10] = ['', ['MinBias','DIGI1','RECO1']]
workflows[12] = ['', ['QCD_Pt_3000_3500','DIGI1','RECO1']]
workflows[14] = ['', ['QCD_Pt_80_120','DIGI1','RECO1']]
workflows[16] = ['', ['SingleElectronPt10','DIGI1','RECO1']]
workflows[17] = ['', ['SingleElectronPt35','DIGI1','RECO1']]
workflows[18] = ['', ['SingleGammaPt10','DIGI1','RECO1']]
workflows[19] = ['', ['SingleGammaPt35','DIGI1','RECO1']]
workflows[20] = ['', ['SingleMuPt10','DIGI1','RECO1']]
workflows[21] = ['', ['SingleMuPt100','DIGI1','RECO1']]
workflows[22] = ['', ['SingleMuPt1000','DIGI1','RECO1']]
workflows[24] = ['', ['TTbar','DIGI1','RECO1']]
workflows[28] = ['', ['ZEE','DIGI1','RECO1']]
workflows[35] = ['', ['Wjet_Pt_80_120','DIGI1','RECO1']]
workflows[36] = ['', ['Wjet_Pt_3000_3500','DIGI1','RECO1']]
workflows[37] = ['', ['LM1_sfts','DIGI1','RECO1']]
workflows[38] = ['', ['QCD_FlatPt_15_3000','DIGI1','RECO1']]

workflows[9]  = ['', ['Higgs200ChargedTaus','DIGI2','RECO2']]
workflows[13] = ['', ['QCD_Pt_3000_3500_2','DIGI2','RECO2']]
workflows[23] = ['', ['JpsiMM','DIGI2','RECO2']]
workflows[25] = ['TTbar', ['TTbar2','DIGI2','RECO2','ALCATT2']]
workflows[26] = ['', ['WE','DIGI2','RECO2']]
workflows[29] = ['ZEE', ['ZEE2','DIGI2','RECO2']]
workflows[31] = ['', ['ZTT','DIGI2','RECO2']]
workflows[32] = ['', ['H130GGgluonfusion','DIGI2','RECO2']]
workflows[33] = ['', ['PhotonJets_Pt_10','DIGI2','RECO2']]
workflows[34] = ['', ['QQH1352T_Tauola','DIGI2','RECO2']]

workflows[7]  = ['', ['Cosmics','DIGICOS','RECOCOS','ALCACOS']]
workflows[8]  = ['', ['BeamHalo','DIGICOS','RECOCOS','ALCABH']]
workflows[11] = ['MinBias', ['MinBias2','DIGI2','RECOMIN','ALCAMIN']]
workflows[15] = ['QCD_Pt_80_120', ['QCD_Pt_80_120_2','DIGI2','RECOQCD','ALCAQCD']]
workflows[27] = ['', ['WM','DIGI2','RECOMU','ALCAMU']]
workflows[30] = ['', ['ZMM','DIGI2','RECOMU','ALCAMU']]


### HI test ###
#workflows[40]
#workflows[41]

workflows[42]=['', ['TTbar_REDIGI_RERECO','REDIGI2RECO','RECOFROMRECO']]

