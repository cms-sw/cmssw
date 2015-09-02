# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

### data ### as it was forgoten in the past 140 + standard wf
workflows[144.5]  = ['', ['RunCosmicsA','RECOCOSD','ALCACOSD']]
workflows[144.45] = ['', ['RunCosmicsA','RECOD']]
workflows[144.6]  = ['', ['MinimumBias2010A','RECOSKIMALCA']]
workflows[144.7]  = ['', ['MinimumBias2010B','RECOSKIMALCA']]
workflows[144.8]  = ['', ['WZMuSkim2010A','RECOSKIMALCA']]
workflows[144.9]  = ['', ['WZEGSkim2010A','RECOSKIMALCA']]
workflows[144.10] = ['', ['WZMuSkim2010B','RECOSKIMALCA']]
workflows[144.11] = ['', ['WZEGSkim2010B','RECOSKIMALCA']]
workflows[144.12] = ['', ['RunMinBias2010B','RECOD']]
workflows[144.13] = ['', ['RunMu2010B','RECOD']]
workflows[144.14] = ['', ['RunElectron2010B','RECOD']]
workflows[144.15] = ['', ['RunPhoton2010B','RECOD']]
workflows[144.16] = ['', ['RunJet2010B','RECOD']]

workflows[101] = [ '', ['SingleElectronE120EHCAL']]
workflows[102] = [ '', ['SinglePiE50HCAL']]

workflows[103]=['',['InclusiveppMuX','DIGI','RECO']]

workflows[105]=['MinBiasHcalNZS',['MinBiasHS','DIGI','ALCANZS']]

workflows[106]=['',['bJpsiX','DIGI','RECO']]
workflows[107]=['',['QQH120Inv','DIGI','RECO']]
workflows[108]=['',['H165WW2L','DIGI','RECO']]
workflows[109]=['',['H200ZZ4L','DIGI','RECO']]
workflows[110]=['',['SingleTauPt50Pythia','DIGI','RECO']]
workflows[111]=['',['JpsiMM_Pt_20_inf','DIGI','RECO']]
workflows[112]=['',['QCD_Pt_15_20','DIGI','RECO']]
workflows[113]=['',['QCD_Pt_20_30','DIGI','RECO']]
workflows[114]=['',['QCD_Pt_30_50','DIGI','RECO']]
workflows[115]=['',['QCD_Pt_50_80','DIGI','RECO']]
workflows[116]=['QCD_Pt_80_120',['QCD_Pt_80_120_2HS','DIGI','RECO']]
workflows[117]=['',['QCD_Pt_120_170','DIGI','RECO']]
workflows[118]=['',['QCD_Pt_170_230','DIGI','RECO']]
workflows[119]=['ZTT',['ZTTHS','DIGI','RECO']]
workflows[120]=['',['SinglePi0E10','DIGI','RECO']]
workflows[121]=['TTbar',['TTbar2HS','DIGI','RECO']]
workflows[122]=['',['UpsMM','DIGI','RECO']]

workflows[123.1]=['QCD_Pt_80_120',['QCD_Pt_80_120FS']]
workflows[123.2]=['QCD_Pt_3000_3500',['QCD_Pt_3000_3500FS']]
workflows[123.3]=['SingleMuPt1',['SingleMuPt1FS']]
workflows[124.1]=['SinglePiPt1',['SinglePiPt1FS']]
workflows[124.2]=['ZTT',['ZTTFS']]
#workflows[124.3]=['ZTT',['ZTTFS2']]
workflows[124.4]=['SinglePiPt10',['SinglePiPt10FS']]
workflows[124.5]=['SinglePiPt100',['SinglePiPt100FS']]
workflows[124.6]=['SingleGammaFlatPt10To10',['SingleGammaFlatPt10To10FS']]

workflows[125]=['',['ZPrime5000Dijet','DIGI','RECO']]
workflows[126]=['',['SingleElectronFlatPt5To100','DIGI','RECO']]
workflows[127]=['',['SingleGammaFlatPt10To100','DIGI','RECO']]
workflows[128]=['',['SingleMuPt1HS','DIGI','RECO']]
workflows[129]=['',['SinglePiPt1','DIGI','RECO']]
workflows[130]=['',['SinglePiPt10','DIGI','RECO']]
workflows[131]=['',['SinglePiPt100','DIGI','RECO']]
workflows[132]=['',['LM9p','DIGI','RECO']]
workflows[133]=['',['RSGrav','DIGI','RECO']]

workflows[134]=['MinimumBiasBS',['RunMinBias2011B','RECOD','ALCAPROMPT','ALCAHARVD']]
