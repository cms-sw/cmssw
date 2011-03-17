
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
workflows[10] = ['', ['MinBias','DIGI','RECO']]
workflows[12] = ['', ['QCD_Pt_3000_3500','DIGI','RECO']]
workflows[14] = ['', ['QCD_Pt_80_120','DIGI','RECO']]
workflows[16] = ['', ['SingleElectronPt10','DIGI','RECO']]
workflows[17] = ['', ['SingleElectronPt35','DIGI','RECO']]
workflows[18] = ['', ['SingleGammaPt10','DIGI','RECO']]
workflows[19] = ['', ['SingleGammaPt35','DIGI','RECO']]
workflows[20] = ['', ['SingleMuPt10','DIGI','RECO']]
workflows[21] = ['', ['SingleMuPt100','DIGI','RECO']]
workflows[22] = ['', ['SingleMuPt1000','DIGI','RECO']]
workflows[24] = ['', ['TTbar','DIGI','RECO']]
workflows[28] = ['', ['ZEE','DIGI','RECO']]
workflows[35] = ['', ['Wjet_Pt_80_120','DIGI','RECO']]
workflows[36] = ['', ['Wjet_Pt_3000_3500','DIGI','RECO']]
workflows[37] = ['', ['LM1_sfts','DIGI','RECO']]
workflows[38] = ['', ['QCD_FlatPt_15_3000','DIGI','RECO']]

workflows[9]  = ['', ['Higgs200ChargedTaus','DIGI','RECO'],stCond]
workflows[13] = ['', ['QCD_Pt_3000_3500_2','DIGI','RECO'],stCond]
workflows[23] = ['', ['JpsiMM','DIGI','RECO'],stCond]
workflows[25] = ['', ['TTbar','DIGI','RECO','ALCATT2'],stCond]
workflows[25.1] = ['TTbar', ['TTbar2','DIGI','RECO','ALCATT2'],stCond]
workflows[26] = ['', ['WE','DIGI','RECO'],stCond]
workflows[29] = ['ZEE', ['ZEE2','DIGI','RECO'],stCond]
workflows[31] = ['', ['ZTT','DIGI','RECO'],stCond]
workflows[32] = ['', ['H130GGgluonfusion','DIGI','RECO'],stCond]
workflows[33] = ['', ['PhotonJets_Pt_10','DIGI','RECO'],stCond]
workflows[34] = ['', ['QQH1352T_Tauola','DIGI','RECO'],stCond]

workflows[7]  = ['', ['Cosmics','DIGICOS','RECOCOS','ALCACOS']]
workflows[8]  = ['', ['BeamHalo','DIGICOS','RECOCOS','ALCABH']]
workflows[11] = ['MinBias', ['MinBias2','DIGI','RECOMIN','ALCAMIN'],stCond]
workflows[15] = ['QCD_Pt_80_120', ['QCD_Pt_80_120_2','DIGI','RECOQCD','ALCAQCD'],stCond]
workflows[27] = ['', ['WM','DIGI','RECOMU','ALCAMU'],stCond]
workflows[30] = ['', ['ZMM','DIGI','RECOMU','ALCAMU'],stCond]


### HI test ###
#workflows[40]
#workflows[41]

workflows[42]=['', ['TTbar_REDIGI_RERECO','REDIGI2RECO','RECOFROMRECO']]

