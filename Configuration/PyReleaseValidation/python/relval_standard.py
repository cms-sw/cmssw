
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

## production tests
workflows[1] = ['', ['ProdMinBias','DIGIPROD1','RECOPROD1']]
workflows[2] = ['', ['ProdTTbar','DIGIPROD1','RECOPROD1']]
workflows[3] = ['', ['ProdQCD_Pt_3000_3500','DIGIPROD1','RECOPROD1']]

### data ###
workflows[4.5]  = ['', ['RunCosmicsA','RECOCOSD','ALCACOSD']]
workflows[4.6]  = ['', ['MinimumBias2010A','RECOSKIM','HARVESTD']]
#workflows[4.7]  = ['', ['MinimumBias2010B','RECOSKIMALCA']]
#workflows[4.8]  = ['', ['WZMuSkim2010A','RECOSKIM']]
#workflows[4.9]  = ['', ['WZEGSkim2010A','RECOSKIM']]
#workflows[4.10] = ['', ['WZMuSkim2010B','RECOSKIM']]
#workflows[4.11] = ['', ['WZEGSkim2010B','RECOSKIM']]

workflows[4.12] = ['', ['RunMinBias2010B','RECOD']]
#workflows[4.13] = ['', ['RunMu2010B','RECOD']]
#workflows[4.14] = ['', ['RunElectron2010B','RECOD']]
#workflows[4.15] = ['', ['RunPhoton2010B','RECOD']]
#workflows[4.16] = ['', ['RunJet2010B','RECOD']]


workflows[4.17] = ['', ['RunMinBias2011A','HLTD','RECODreHLT','HARVESTDreHLT','SKIMDreHLT']]
workflows[4.18] = ['', ['RunMu2011A','RECOD']]
workflows[4.19] = ['', ['RunElectron2011A','RECOD']]
workflows[4.20] = ['', ['RunPhoton2011A','RECOD']]
workflows[4.21] = ['', ['RunJet2011A','RECOD']]

workflows[4.22] = ['', ['RunCosmics2011A','RECOCOSD','ALCACOSD','SKIMCOSD','HARVESTDC']]

workflows[4.23] = ['',['ValSkim2011A','RECOSKIM']]
workflows[4.24] = ['',['WMuSkim2011A','RECOSKIM']]
workflows[4.25] = ['',['WElSkim2011A','RECOSKIM']]
workflows[4.26] = ['',['ZMuSkim2011A','RECOSKIM']]
workflows[4.27] = ['',['ZElSkim2011A','RECOSKIM']]
workflows[4.28] = ['',['HighMet2011A','RECOSKIM']]

workflows[4.29] = ['', ['RunMinBias2011B','HLTD','RECODreHLT','HARVESTDreHLT','SKIMDreHLT']]
#workflows[4.291] = ['', ['RunMinBias2011B','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.30] = ['', ['RunMu2011B','HLTD','RECODreHLT']]
workflows[4.31] = ['', ['RunElectron2011B','HLTD','RECODreHLT']]
workflows[4.32] = ['', ['RunPhoton2011B','HLTD','RECODreHLT']]
workflows[4.33] = ['', ['RunJet2011B','HLTD','RECODreHLT']]

workflows[4.34] = ['',['ValSkim2011B','RECOSKIM']]
workflows[4.35] = ['',['WMuSkim2011B','RECOSKIM']]
workflows[4.36] = ['',['WElSkim2011B','RECOSKIM']]
workflows[4.37] = ['',['ZMuSkim2011B','RECOSKIM']]
workflows[4.38] = ['',['ZElSkim2011B','RECOSKIM']]
workflows[4.39] = ['',['HighMet2011B','RECOSKIM']]

workflows[4.40] = ['',['RunMinBias2012A','HLTD','RECODreHLT']]
workflows[4.41] = ['',['RunTau2012A','HLTD','RECODreHLT']]
workflows[4.42] = ['',['RunMET2012A','HLTD','RECODreHLT']]
workflows[4.43] = ['',['RunMu2012A','HLTD','RECODreHLT']]
workflows[4.44] = ['',['RunElectron2012A','HLTD','RECODreHLT']]
workflows[4.45] = ['',['RunJet2012A','HLTD','RECODreHLT']]

workflows[4.51] = ['',['RunMinBias2012B','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.52] = ['',['RunMu2012B','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.53] = ['',['RunPhoton2012B','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.54] = ['',['RunEl2012B','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.55] = ['',['RunJet2012B','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.56] = ['',['ZMuSkim2012B','RECOSKIM']]
workflows[4.57] = ['',['ZElSkim2012B','RECOSKIM']]
workflows[4.58] = ['',['WElSkim2012B','RECOSKIM']]

workflows[4.61] = ['',['RunMinBias2012C','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.62] = ['',['RunMu2012C','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.63] = ['',['RunPhoton2012C','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.64] = ['',['RunEl2012C','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.65] = ['',['RunJet2012C','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.66] = ['',['ZMuSkim2012C','RECOSKIM','HARVESTD']]
workflows[4.67] = ['',['ZElSkim2012C','RECOSKIM','HARVESTD']]
workflows[4.68] = ['',['WElSkim2012C','RECOSKIM','HARVESTD']]

workflows[4.71] = ['',['RunMinBias2012D','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.72] = ['',['RunMu2012D','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.73] = ['',['RunPhoton2012D','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.74] = ['',['RunEl2012D','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.75] = ['',['RunJet2012D','HLTD','RECODreHLT','HARVESTDreHLT']]
workflows[4.76] = ['',['ZMuSkim2012D','RECOSKIM','HARVESTD']]
workflows[4.77] = ['',['ZElSkim2012D','RECOSKIM','HARVESTD']]
workflows[4.78] = ['',['WElSkim2012D','RECOSKIM','HARVESTD']]

workflows[40.51] = ['',['RunHI2010','REPACKHID','RECOHID11St3']]
workflows[40.52] = ['',['RunHI2010','RECOHID10','RECOHIR10D11']]
workflows[40.53] = ['',['RunHI2011','RECOHID11','HARVESTDHI']]

### fastsim ###
workflows[5.1] = ['TTbar', ['TTbarFS','HARVESTFS']]
workflows[5.2] = ['SingleMuPt10', ['SingleMuPt10FS']]
workflows[5.3] = ['SingleMuPt100', ['SingleMuPt100FS']]
workflows[5.4] = ['ZEE', ['ZEEFS']]
workflows[5.5] = ['ZTT',['ZTTFS']]

workflows[39]  = ['QCD_FlatPt_15_3000', ['QCDFlatPt153000FS']]
workflows[6.4] = ['H130GGgluonfusion', ['H130GGgluonfusionFS']]

### standard set ###
#workflows[10] = ['', ['MinBias','DIGI','RECO']]
#workflows[12] = ['', ['QCD_Pt_3000_3500','DIGI','RECO']]
#workflows[14] = ['', ['QCD_Pt_80_120','DIGI','RECO']]
workflows[15] = ['', ['SingleElectronPt10','DIGI','RECO']]
workflows[16] = ['', ['SingleElectronPt1000','DIGI','RECO']]
workflows[17] = ['', ['SingleElectronPt35','DIGI','RECO']]
workflows[18] = ['', ['SingleGammaPt10','DIGI','RECO']]
workflows[19] = ['', ['SingleGammaPt35','DIGI','RECO']]
workflows[20] = ['', ['SingleMuPt10','DIGI','RECO']]
workflows[21] = ['', ['SingleMuPt100','DIGI','RECO']]
workflows[22] = ['', ['SingleMuPt1000','DIGI','RECO']]
workflows[24] = ['', ['TTbarLepton','DIGI','RECO','HARVEST']]
#workflows[28] = ['', ['ZEE','DIGI','RECO']]
workflows[35] = ['', ['Wjet_Pt_80_120','DIGI','RECO']]
workflows[36] = ['', ['Wjet_Pt_3000_3500','DIGI','RECO']]
workflows[37] = ['', ['LM1_sfts','DIGI','RECO']]
workflows[38] = ['', ['QCD_FlatPt_15_3000','DIGI','RECO']]

workflows[9]  = ['', ['Higgs200ChargedTaus','DIGI','RECO']]
workflows[13] = ['', ['QCD_Pt_3000_3500','DIGI','RECO']]
workflows[23] = ['', ['JpsiMM','DIGI','RECO']]
workflows[25] = ['', ['TTbar','DIGI','RECO','ALCATT']]
workflows[26] = ['', ['WE','DIGI','RECO','HARVEST']]
workflows[29] = ['', ['ZEE','DIGI','RECO','ALCAELE']]
workflows[31] = ['', ['ZTT','DIGI','RECO']]
workflows[32] = ['', ['H130GGgluonfusion','DIGI','RECO']]
workflows[33] = ['', ['PhotonJets_Pt_10','DIGI','RECO']]
workflows[34] = ['', ['QQH1352T_Tauola','DIGI','RECO']]

workflows[7]  = ['', ['Cosmics','DIGICOS','RECOCOS','ALCACOS']]
workflows[8]  = ['', ['BeamHalo','DIGICOS','RECOCOS','ALCABH','HARVESTCOS']]
workflows[11] = ['', ['MinBias','DIGI','RECOMIN','ALCAMIN']]
workflows[28] = ['', ['QCD_Pt_80_120','DIGI','RECO']]
workflows[27] = ['', ['WM','DIGI','RECO']]
workflows[30] = ['', ['ZMM','DIGI','RECO']]


### HI test ###
workflows[40] = ['',['HydjetQ_MinBias_2760GeV','DIGIHI','RECOHI','HARVESTHI']]
workflows[41] = ['',['HydjetQ_B0_2760GeV','DIGIHI','RECOHI']]
workflows[42] = ['',['HydjetQ_B8_2760GeV','DIGIHI','RECOHI']]

### pPb test ###
workflows[80]= ['',['Hijing_PPb_5020GeV_MinimumBias','DIGI','RECO']]

#  LocalWords:  workflows
