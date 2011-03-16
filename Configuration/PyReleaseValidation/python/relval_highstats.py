# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = {}

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

### data ### as it was forgoten in the past 140 + standard wf
workflows[144.5]  = ['', ['RunCosmicsA','RECOCOSD','ALCACOSD']]
workflows[144.45] = ['', ['RunCosmicsA','RECOD']]
workflows[144.6]  = ['', ['MinimumBias2010A','RECOVALSKIM']]
workflows[144.7]  = ['', ['MinimumBias2010B','RECOVALSKIMALCA']]
workflows[144.8]  = ['', ['WZMuSkim2010A','RECOVALSKIM']]
workflows[144.9]  = ['', ['WZEGSkim2010A','RECOVALSKIM']]
workflows[144.10] = ['', ['WZMuSkim2010B','RECOVALSKIM']]
workflows[144.11] = ['', ['WZEGSkim2010B','RECOVALSKIM']]
workflows[144.12] = ['', ['RunMinBias2010B','RECOD']]
workflows[144.13] = ['', ['RunMu2010B','RECOD']]
workflows[144.14] = ['', ['RunElectron2010B','RECOD']]
workflows[144.15] = ['', ['RunPhoton2010B','RECOD']]
workflows[144.16] = ['', ['RunJet2010B','RECOD']]

workflows[101] = [ '', ['SingleElectronE120EHCAL']]
workflows[102] = [ '', ['SinglePiE50HCAL']]

workflows[103]=['',['InclusiveppMuX','DIGI1','RECO1']]

workflows[105]=['MinBiasHcalNZS',['MinBiasHS','ALCANZS']]

workflows[106]=['',['bJpsiX','DIGI2','RECO2']]
workflows[107]=['',['QQH120Inv','DIGI2','RECO2']]
workflows[108]=['',['H165WW2L','DIGI2','RECO2']]
workflows[109]=['',['H200ZZ4L','DIGI2','RECO2']]
workflows[110]=['',['SingleTauPt50Pythia','DIGI1','RECO1']]
workflows[111]=['',['JpsiMM_Pt_20_inf','DIGI2','RECO2']]
workflows[112]=['',['QCD_Pt_15_20','DIGI2','RECO2']]
workflows[113]=['',['QCD_Pt_20_30','DIGI2','RECO2']]
workflows[114]=['',['QCD_Pt_30_50','DIGI2','RECO2']]
workflows[115]=['',['QCD_Pt_50_80','DIGI2','RECO2']]
workflows[116]=['QCD_Pt_80_120',['QCD_Pt_80_120_2HS','DIGI2','RECO2']]
workflows[117]=['',['QCD_Pt_120_170','DIGI2','RECO2']]
workflows[118]=['',['QCD_Pt_170_230','DIGI2','RECO2']]
workflows[119]=['ZTT',['ZTTHS','DIGI2','RECO2']]
workflows[120]=['',['SinglePi0E10','DIGI1','RECO1']]
workflows[121]=['TTbar',['TTbar2HS','DIGI2','RECO2']]
workflows[122]=['',['UpsMM','DIGI2','RECO2']]

#workflows[123.1]
#workflows[123.2]
#workflows[123.3]
#workflows[124.1]
#workflows[124.2]
#workflows[124.3]
#workflows[124.4]
#workflows[124.5]
#workflows[124.6]

workflows[125]=['',['ZPrime5000Dijet','DIGI1','RECO1']]
workflows[126]=['',['SingleElectronFlatPt5To100','DIGI1','RECO1']]
workflows[127]=['',['SingleGammaFlatPt10To100','DIGI1','RECO1']]
workflows[128]=['',['SingleMuPt1','DIGI1','RECO1']]
workflows[129]=['',['SinglePiPt1','DIGI1','RECO1']]
workflows[130]=['',['SinglePiPt10','DIGI1','RECO1']]
workflows[131]=['',['SinglePiPt100','DIGI1','RECO1']]
workflows[132]=['',['LM9p','DIGI2','RECO2']]
workflows[133]=['',['RSGrav','DIGI2','RECO2']]

#workflows[134] #dropped
