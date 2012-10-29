
InputInfoNDefault=2000000    
class InputInfo(object):
    def __init__(self,dataSet,label='',run=0,files=1000,events=InputInfoNDefault,location='CAF') :
        self.run = run
        self.files = files
        self.events = events
        self.location = location
        self.label = label
        self.dataSet = dataSet

# merge dictionaries, with prioty on the [0] index
def merge(dictlist,TELL=False):
    import copy
    last=len(dictlist)-1
    if TELL: print last,dictlist
    if last==0:
        # ONLY ONE ITEM LEFT
        return copy.copy(dictlist[0])
    else:
        reducedlist=dictlist[0:max(0,last-1)]
        if TELL: print reducedlist
        # make a copy of the last item
        d=copy.copy(dictlist[last])
        # update with the last but one item
        d.update(dictlist[last-1])
        # and recursively do the rest
        reducedlist.append(d)
        return merge(reducedlist,TELL)


# step1 gensim
step1Defaults = {'--relval'      : None, # need to be explicitly set
                 '-s'            : 'GEN,SIM',
                 '-n'            : 10,
                 '--conditions'  : 'auto:startup',
                 '--datatier'    : 'GEN-SIM',
                 '--eventcontent': 'RAWSIM',
                 }

step1 = {}

#### Production test section ####
step1['ProdMinBias']=merge([{'cfg':'MinBias_7TeV_cfi','--relval':'9000,100'},step1Defaults])
step1['ProdTTbar']=merge([{'cfg':'TTbar_Tauola_7TeV_cfi','--relval':'9000,50'},step1Defaults])
step1['ProdQCD_Pt_3000_3500']=merge([{'cfg':'QCD_Pt_3000_3500_7TeV_cfi','--relval':'9000,25'},step1Defaults])

#step1['ProdMinBiasINPUT']={'INPUT':InputInfo(dataSet='/RelValProdMinBias/CMSSW_4_3_0_pre2-MC_42_V9-v1/GEN-SIM',label='prodmbrv',location='STD')}
#step1['ProdTTbarINPUT']={'INPUT':InputInfo(dataSet='/RelValProdTTbar/CMSSW_4_3_0_pre2-MC_42_V9-v1/GEN-SIM',label='prodttbrv',location='STD')}
#step1['ProdQCD_Pt_3000_3500INPUT']={'INPUT':InputInfo(dataSet='/RelValProdQCD_Pt_3000_3500/CMSSW_4_3_0_pre2-MC_42_V9-v1/GEN-SIM',label='qcd335',location='STD')}



#### data ####
step1['MinimumBias2010A']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2010A-valskim-v6/RAW-RECO',label='run2010A',location='STD')}
step1['MinimumBias2010B']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2010B-valskim-v2/RAW-RECO',label='run2010B')}
step1['WZMuSkim2010A']={'INPUT':InputInfo(dataSet='/Mu/Run2010A-WZMu-Nov4Skim_v1/RAW-RECO',label='wzMu2010A')}
step1['WZMuSkim2010B']={'INPUT':InputInfo(dataSet='/Mu/Run2010B-WZMu-Nov4Skim_v1/RAW-RECO',label='wzMu2010B')}
step1['WZEGSkim2010A']={'INPUT':InputInfo(dataSet='/EG/Run2010A-WZEG-Nov4Skim_v1/RAW-RECO',label='wzEG2010A')}
step1['WZEGSkim2010B']={'INPUT':InputInfo(dataSet='/Electron/Run2010B-WZEG-Nov4Skim_v1/RAW-RECO',label='wzEG2010B')}

step1['RunCosmicsA']={'INPUT':InputInfo(dataSet='/Cosmics/Run2010A-v1/RAW',label='cos2010A',run=142089,events=100000)}
Run2010B=149011
step1['RunMinBias2010B']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2010B-RelValRawSkim-v1/RAW',label='mb2010B',run=Run2010B,events=100000)}
step1['RunMu2010B']={'INPUT':InputInfo(dataSet='/Mu/Run2010B-RelValRawSkim-v1/RAW',label='mu2010B',run=Run2010B,events=100000)}
step1['RunElectron2010B']={'INPUT':InputInfo(dataSet='/Electron/Run2010B-RelValRawSkim-v1/RAW',label='electron2010B',run=Run2010B,events=100000)}
step1['RunPhoton2010B']={'INPUT':InputInfo(dataSet='/Photon/Run2010B-RelValRawSkim-v1/RAW',label='photon2010B',run=Run2010B,events=100000)}
step1['RunJet2010B']={'INPUT':InputInfo(dataSet='/Jet/Run2010B-RelValRawSkim-v1/RAW',label='jet2010B',run=Run2010B,events=100000)}

Run2011ASk=172802
step1['ValSkim2011A']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2011A-ValSkim-PromptSkim-v6/RAW-RECO',label='run2011A',location='STD',run=Run2011ASk)}
step1['WMuSkim2011A']={'INPUT':InputInfo(dataSet='/SingleMu/Run2011A-WMu-PromptSkim-v6/RAW-RECO',label='wMu2011A',location='STD',run=Run2011ASk)}
step1['WElSkim2011A']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2011A-WElectron-PromptSkim-v6/RAW-RECO',label='wEl2011A',location='STD',run=Run2011ASk)}
step1['ZMuSkim2011A']={'INPUT':InputInfo(dataSet='/DoubleMu/Run2011A-ZMu-PromptSkim-v6/RAW-RECO',label='zMu2011A',location='STD',run=Run2011ASk)}
step1['ZElSkim2011A']={'INPUT':InputInfo(dataSet='/DoubleElectron/Run2011A-ZElectron-PromptSkim-v6/RAW-RECO',label='zEl2011A',location='STD',run=Run2011ASk)}
step1['HighMet2011A']={'INPUT':InputInfo(dataSet='/Jet/Run2011A-HighMET-PromptSkim-v6/RAW-RECO',label='hMet2011A',location='STD',run=Run2011ASk)}

step1['RunCosmics2011A']={'INPUT':InputInfo(dataSet='/Cosmics/Run2011A-v1/RAW',label='cos2011A',run=160960,events=100000,location='STD')}
Run2011A=165121
step1['RunMinBias2011A']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2011A-v1/RAW',label='mb2011A',run=Run2011A,events=100000,location='STD')}
step1['RunMu2011A']={'INPUT':InputInfo(dataSet='/SingleMu/Run2011A-v1/RAW',label='mu2011A',run=Run2011A,events=100000)}
step1['RunElectron2011A']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2011A-v1/RAW',label='electron2011A',run=Run2011A,events=100000)}
step1['RunPhoton2011A']={'INPUT':InputInfo(dataSet='/Photon/Run2011A-v1/RAW',label='photon2011A',run=Run2011A,events=100000)}
step1['RunJet2011A']={'INPUT':InputInfo(dataSet='/Jet/Run2011A-v1/RAW',label='jet2011A',run=Run2011A,events=100000)}

step1['RunHI2010']={'INPUT':InputInfo(dataSet='/HIAllPhysics/HIRun2010-v1/RAW',label='hi2010',run=152957,events=100000,location='STD')}
step1['RunHI2011']={'INPUT':InputInfo(dataSet='/HIAllPhysics/HIRun2011A-v1/RAW',label='hi2011',run=174773,events=100000,location='STD')}

#### Standard release validation samples ####

stCond={'--conditions':'auto:startup'}
K9by25={'--relval':'9000,25'}
K9by50={'--relval':'9000,50'}
K9by100={'--relval':'9000,100'}
K9by250={'--relval':'9000,250'}
K25by250={'--relval':'25000,250'}

def gen(fragment,howMuch):
    global step1Defaults
    return merge([{'cfg':fragment},howMuch,step1Defaults])

step1['MinBias']=gen('MinBias_7TeV_cfi',K9by100)
step1['QCD_Pt_3000_3500']=gen('QCD_Pt_3000_3500_7TeV_cfi',K9by25)
step1['QCD_Pt_80_120']=gen('QCD_Pt_80_120_7TeV_cfi',K9by50)
step1['SingleElectronPt10']=gen('SingleElectronPt10_cfi',K9by250)
step1['SingleElectronPt35']=gen('SingleElectronPt35_cfi',K9by250)
step1['SingleGammaPt10']=gen('SingleGammaPt10_cfi',K9by100)
step1['SingleGammaPt35']=gen('SingleGammaPt35_cfi',K9by100)
step1['SingleMuPt10']=gen('SingleMuPt10_cfi',K25by250)
step1['SingleMuPt100']=gen('SingleMuPt100_cfi',K9by250)
step1['SingleMuPt1000']=gen('SingleMuPt1000_cfi',K9by100)
step1['TTbar']=gen('TTbar_Tauola_7TeV_cfi',K9by50)
step1['ZEE']=gen('ZEE_7TeV_cfi',K9by100)
step1['Wjet_Pt_80_120']=gen('Wjet_Pt_80_120_7TeV_cfi',K9by100)
step1['Wjet_Pt_3000_3500']=gen('Wjet_Pt_3000_3500_7TeV_cfi',K9by100)
step1['LM1_sfts']=gen('LM1_sfts_7TeV_cfi',K9by100)
step1['QCD_FlatPt_15_3000']=gen('QCDForPF_7TeV_cfi',K9by100)

step1['MinBiasINPUT']={'INPUT':InputInfo(dataSet='/RelValMinBias/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['QCD_Pt_3000_3500INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_3000_3500/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['QCD_Pt_80_120INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_80_120/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['SingleElectronPt10INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleElectronPt10/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['SingleElectronPt35INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleElectronPt35/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['SingleGammaPt10INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleGammaPt10/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['SingleGammaPt35INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleGammaPt35/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['SingleMuPt10INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt10/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['SingleMuPt100INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt100/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['SingleMuPt1000INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt1000/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['TTbarINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['OldTTbarINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/CMSSW_4_2_6-START42_V12-v1/GEN-SIM-RECO',location='STD')}
step1['ZEEINPUT']={'INPUT':InputInfo(dataSet='/RelValZEE/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['Wjet_Pt_80_120INPUT']={'INPUT':InputInfo(dataSet='/RelValWjet_Pt_80_120/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['Wjet_Pt_3000_3500INPUT']={'INPUT':InputInfo(dataSet='/RelValWjet_Pt_3000_3500/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['LM1_sftsINPUT']={'INPUT':InputInfo(dataSet='/RelValLM1_sfts/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['QCD_FlatPt_15_3000INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_FlatPt_15_3000/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}

## high stat step1
K700by280={'--relval': '70000,280'}
K250by100={'--relval': '25000,100'}
K3250000by1300000={'--relval': '325000000,1300000'}
K250by250={'--relval': '25000,250'}
K110000by45000={'--relval': '11000000,45000'}
K562by225={'--relval': '56250,225'}

ecalHcal={
    '-s':'GEN,SIM,DIGI,DIGI2RAW,RAW2DIGI,L1Reco,RECO',
    '--datatier':'GEN-SIM-DIGI-RAW-RECO',
    #'--geometry':'ECALHCAL',
    '--eventcontent':'FEVTDEBUG',
    '--customise':'Validation/Configuration/ECALHCAL.customise',
    '--beamspot':'NoSmear'}
step1['SingleElectronE120EHCAL']=merge([{'cfg':'SingleElectronE120EHCAL_cfi'},ecalHcal,K25by250,step1Defaults])
step1['SinglePiE50HCAL']=merge([{'cfg':'SinglePiE50HCAL_cfi'},ecalHcal,K25by250,step1Defaults])

step1['MinBiasHS']=gen('MinBias_7TeV_cfi',K25by250)
step1['InclusiveppMuX']=gen('InclusiveppMuX_7TeV_cfi',K110000by45000)
step1['SingleElectronFlatPt5To100']=gen('SingleElectronFlatPt5To100_cfi',K250by250)
step1['SinglePiPt1']=gen('SinglePiPt1_cfi',K250by250)
step1['SingleMuPt1']=gen('SingleMuPt1_cfi',K250by250)
step1['ZPrime5000Dijet']=gen('ZPrime5000JJ_7TeV_cfi',K250by100)
step1['SinglePi0E10']=gen('SinglePi0E10_cfi',K250by100)
step1['SinglePiPt10']=gen('SinglePiPt10_cfi',K250by250)
step1['SingleGammaFlatPt10To100']=gen('SingleGammaFlatPt10To100_cfi',K250by250)
step1['SingleTauPt50Pythia']=gen('SingleTaupt_50_cfi',K250by100)
step1['SinglePiPt100']=gen('SinglePiPt100_cfi',K250by250)


def genS(fragment,howMuch):
    global step1Defaults,stCond
    return merge([{'cfg':fragment},stCond,howMuch,step1Defaults])

##step1['MinBias2']=genS('MinBias_7TeV_cfi',K9by100)
step1['Higgs200ChargedTaus']=genS('H200ChargedTaus_Tauola_7TeV_cfi',K9by100)
##step1['QCD_Pt_3000_3500_2']=genS('QCD_Pt_3000_3500_7TeV_cfi',K9by25)
##step1['QCD_Pt_80_120_2']=genS('QCD_Pt_80_120_7TeV_cfi',K9by50)
step1['JpsiMM']=genS('JpsiMM_7TeV_cfi',{'--relval':'65250,725'})
##step1['TTbar2']=genS('TTbar_Tauola_7TeV_cfi',K9by50)
step1['WE']=genS('WE_7TeV_cfi',K9by100)
step1['WM']=genS('WM_7TeV_cfi',K9by100)
##step1['ZEE2']=genS('ZEE_7TeV_cfi',K9by100)
step1['ZMM']=genS('ZMM_7TeV_cfi',{'--relval':'18000,200'})
step1['ZTT']=genS('ZTT_Tauola_All_hadronic_7TeV_cfi',K9by100)
step1['H130GGgluonfusion']=genS('H130GGgluonfusion_7TeV_cfi',K9by100)
step1['PhotonJets_Pt_10']=genS('PhotonJet_Pt_10_7TeV_cfi',K9by100)
step1['QQH1352T_Tauola']=genS('QQH1352T_Tauola_7TeV_cfi',K9by100)

step1['MinBias2INPUT']={'INPUT':InputInfo(dataSet='/RelValMinBias/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['Higgs200ChargedTausINPUT']={'INPUT':InputInfo(dataSet='/RelValHiggs200ChargedTaus/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['QCD_Pt_3000_3500_2INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_3000_3500/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['QCD_Pt_80_120_2INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_80_120/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['JpsiMMINPUT']={'INPUT':InputInfo(dataSet='/RelValJpsiMM/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['TTbar2INPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['WEINPUT']={'INPUT':InputInfo(dataSet='/RelValWE/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['WMINPUT']={'INPUT':InputInfo(dataSet='/RelValWM/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['ZEE2INPUT']={'INPUT':InputInfo(dataSet='/RelValZEE/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['ZMMINPUT']={'INPUT':InputInfo(dataSet='/RelValZMM/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['ZTTINPUT']={'INPUT':InputInfo(dataSet='/RelValZTT/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['H130GGgluonfusionINPUT']={'INPUT':InputInfo(dataSet='/RelValH130GGgluonfusion/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['PhotonJets_Pt_10INPUT']={'INPUT':InputInfo(dataSet='/RelValPhotonJets_Pt_10/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['QQH1352T_TauolaINPUT']={'INPUT':InputInfo(dataSet='/RelValQQH1352T_Tauola/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}

step1['Cosmics']=merge([{'cfg':'UndergroundCosmicMu_cfi.py','--relval':'666000,7400','--scenario':'cosmics'},step1Defaults])
step1['BeamHalo']=merge([{'cfg':'BeamHalo_cfi.py','--scenario':'cosmics'},K9by100,step1Defaults])

step1['CosmicsINPUT']={'INPUT':InputInfo(dataSet='/RelValCosmics/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}
step1['BeamHaloINPUT']={'INPUT':InputInfo(dataSet='/RelValBeamHalo/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM',location='STD')}

step1['QCD_Pt_50_80']=genS('QCD_Pt_50_80_7TeV_cfi',K250by100)
step1['QCD_Pt_15_20']=genS('QCD_Pt_15_20_7TeV_cfi',K250by100)
step1['ZTTHS']=merge([K250by100,step1['ZTT']])
step1['QQH120Inv']=genS('QQH120Inv_7TeV_cfi',K250by100)
step1['TTbar2HS']=merge([K250by100,step1['TTbar']])
step1['JpsiMM_Pt_20_inf']=genS('JpsiMM_Pt_20_inf_7TeV_cfi',K700by280)
step1['QCD_Pt_120_170']=genS('QCD_Pt_120_170_7TeV_cfi',K250by100)
step1['H165WW2L']=genS('H165WW2L_Tauola_7TeV_cfi',K250by100)
step1['UpsMM']=genS('UpsMM_7TeV_cfi',K562by225)
step1['RSGrav']=genS('RS750_quarks_and_leptons_7TeV_cff',K250by100)
step1['QCD_Pt_80_120_2HS']=merge([K250by100,step1['QCD_Pt_80_120']])
step1['bJpsiX']=genS('bJpsiX_7TeV_cfi',K3250000by1300000)
step1['QCD_Pt_30_50']=genS('QCD_Pt_30_50_7TeV_cfi',K250by100)
step1['H200ZZ4L']=genS('H200ZZ4L_Tauola_7TeV_cfi',K250by100)
step1['LM9p']=genS('LM9p_7TeV_cff',K250by100)
step1['QCD_Pt_20_30']=genS('QCD_Pt_20_30_7TeV_cfi',K250by100)
step1['QCD_Pt_170_230']=genS('QCD_Pt_170_230_7TeV_cfi',K250by100)

## heavy ions tests
U500by5={'--relval': '500,5'}
U80by2={'--relval': '80,2'}
hiDefaults={'--conditions':'auto:starthi',
           '--scenario':'HeavyIons'}

step1['HydjetQ_MinBias_2760GeV']=merge([{'-n':1},hiDefaults,genS('Hydjet_Quenched_MinBias_2760GeV_cfi',U500by5)])
step1['HydjetQ_MinBias_2760GeVINPUT']={'INPUT':InputInfo(dataSet='/RelValHydjetQ_MinBias_2760GeV/CMSSW_4_4_2_patch10-STARTHI44_V6_special_120119-v1/GEN-SIM',location='STD')}
step1['HydjetQ_B0_2760GeV']=merge([{'-n':1},hiDefaults,genS('Hydjet_Quenched_B0_2760GeV_cfi',U80by2)])
step1['HydjetQ_B0_2760GeVINPUT']={'INPUT':InputInfo(dataSet='/RelValHydjetQ_B0_2760GeV/CMSSW_4_4_2_patch10-STARTHI44_V6_special_120119-v1/GEN-SIM')}
step1['HydjetQ_B0_2760GeVPUINPUT']={'INPUT':InputInfo(dataSet='/RelValHydjetQ_B0_2760GeV/CMSSW_4_4_2_patch10-STARTHI44_V6_special_120119-v1/GEN-SIM')}
step1['HydjetQ_B8_2760GeV']=merge([{'-n':1},hiDefaults,genS('Hydjet_Quenched_B8_2760GeV_cfi',U80by2)])
step1['HydjetQ_B8_2760GeVINPUT']={'INPUT':InputInfo(dataSet='/RelValHydjetQ_B8_2760GeV/CMSSW_4_4_2_patch10-STARTHI44_V6_special_120119-v1/GEN-SIM',location='STD')}




def changeRefRelease(step1s,listOfPairs):
    for s in step1s:
        if ('INPUT' in step1s[s]):
            oldD=step1[s]['INPUT'].dataSet
            for ref,newRef in listOfPairs:
                if  ref in oldD:
                    step1[s]['INPUT'].dataSet=oldD.replace(ref,newRef)
                                        
def addForAll(steps,d):
    for s in steps:
        steps[s].update(d)


#changeRefRelease(step1,[('CMSSW_4_3_0_pre3-START43_V1-v2','CMSSW_4_3_0_pre3-START43_V1-v2'),
#                        ('CMSSW_4_3_0_pre3-MC_43_V1-v2','CMSSW_4_3_0_pre3-MC_43_V1-v2')
#                        ])
                        

#### fastsim section ####
##no forseen to do things in two steps GEN-SIM then FASTIM->end: maybe later
step1FastDefaults =merge([{'-s':'GEN,FASTSIM,HLT:GRun,VALIDATION',
                           '--eventcontent':'FEVTDEBUGHLT',
                           '--datatier':'GEN-SIM-DIGI-RECO',
                           '--relval':'27000,1000'},
                          step1Defaults])
K100byK1={'--relval':'100000,1000'}
step1['TTbarFS']=merge([{'cfg':'TTbar_Tauola_7TeV_cfi'},K100byK1,step1FastDefaults])
#step1['TTbarFS2']=merge([{'cfg':'TTbar_Tauola_7TeV_cfi'},K100byK1,stCond,step1FastDefaults])
step1['SingleMuPt1FS']=merge([{'cfg':'SingleMuPt1_cfi'},step1FastDefaults])
step1['SingleMuPt10FS']=merge([{'cfg':'SingleMuPt10_cfi'},step1FastDefaults])
step1['SingleMuPt100FS']=merge([{'cfg':'SingleMuPt100_cfi'},step1FastDefaults])
step1['SinglePiPt1FS']=merge([{'cfg':'SinglePiPt1_cfi'},step1FastDefaults])
step1['SinglePiPt10FS']=merge([{'cfg':'SinglePiPt10_cfi'},step1FastDefaults])
step1['SinglePiPt100FS']=merge([{'cfg':'SinglePiPt100_cfi'},step1FastDefaults])
step1['ZEEFS']=merge([{'cfg':'ZEE_7TeV_cfi'},K100byK1,step1FastDefaults])
#step1['ZEEFS2']=merge([{'cfg':'ZEE_7TeV_cfi'},K100byK1,stCond,step1FastDefaults])
step1['ZTTFS']=merge([{'cfg':'ZTT_Tauola_OneLepton_OtherHadrons_7TeV_cfi'},K100byK1,step1FastDefaults])
#step1['ZTTFS2']=merge([{'cfg':'ZTT_Tauola_OneLepton_OtherHadrons_7TeV_cfi'},K100byK1,stCond,step1FastDefaults])
step1['QCDFlatPt153000FS']=merge([{'cfg':'QCDForPF_7TeV_cfi'},step1FastDefaults])
step1['QCD_Pt_80_120FS']=merge([{'cfg':'QCD_Pt_80_120_7TeV_cfi'},K100byK1,stCond,step1FastDefaults])
step1['QCD_Pt_3000_3500FS']=merge([{'cfg':'QCD_Pt_3000_3500_7TeV_cfi'},K100byK1,stCond,step1FastDefaults])
step1['H130GGgluonfusionFS']=merge([{'cfg':'H130GGgluonfusion_7TeV_cfi'},step1FastDefaults])
step1['SingleGammaFlatPt10To10FS']=merge([{'cfg':'SingleGammaFlatPt10To100_cfi'},K100byK1,step1FastDefaults])

#### generator test section ####
step1GenDefaults=merge([{'-s':'GEN,VALIDATION:genvalid',
                         '--relval':'1000000,20000',
                         '--eventcontent':'RAWSIM',
                         '--datatier':'GEN'},
                        step1Defaults])
def genvalid(fragment,d,suffix='',fi=''):
    import copy
    c=copy.copy(d)
    if suffix:
        c['-s']=c['-s'].replace('genvalid','genvalid_'+suffix)
    if fi:
        c['--filein']='lhe:%d'%(fi,)
    c['cfg']=fragment
    return c
    
step1['QCD_Pt-30_7TeV_herwigpp']=genvalid('QCD_Pt_30_7TeV_herwigpp_cff',step1GenDefaults,'qcd')
step1['DYToLL_M-50_TuneZ2_7TeV_pythia6-tauola']=genvalid('DYToLL_M_50_TuneZ2_7TeV_pythia6_tauola_cff',step1GenDefaults,'dy')
step1['QCD_Pt-30_TuneZ2_7TeV_pythia6']=genvalid('QCD_Pt_30_TuneZ2_7TeV_pythia6_cff',step1GenDefaults,'qcd')
step1['QCD_Pt-30_7TeV_pythia8']=genvalid('QCD_Pt_30_7TeV_pythia8_cff',step1GenDefaults,'qcd')
step1['GluGluTo2Jets_M-100_7TeV_exhume']=genvalid('GluGluTo2Jets_M_100_7TeV_exhume_cff',step1GenDefaults,'qcd')
step1['TT_TuneZ2_7TeV_pythia6-evtgen']=genvalid('TT_TuneZ2_7TeV_pythia6_evtgen_cff',step1GenDefaults)
step1['MinBias_TuneZ2_7TeV_pythia6']=genvalid('MinBias_TuneZ2_7TeV_pythia6_cff',step1GenDefaults,'qcd')
step1['WToLNu_TuneZ2_7TeV_pythia6-tauola']=genvalid('WToLNu_TuneZ2_7TeV_pythia6_tauola_cff',step1GenDefaults,'w')
step1['QCD_Pt-30_7TeV_herwig6']=genvalid('QCD_Pt_30_7TeV_herwig6_cff',step1GenDefaults,'qcd')
step1['MinBias_7TeV_pythia8']=genvalid('MinBias_7TeV_pythia8_cff',step1GenDefaults,'qcd')

step1['DYToMuMu_M-20_7TeV_mcatnlo']=genvalid('DYToMuMu_M_20_7TeV_mcatnlo_cff',step1GenDefaults,'dy',664)
step1['TT_7TeV_mcatnlo']=genvalid('TT_7TeV_mcatnlo_cff',step1GenDefaults,'',346)
step1['WminusToENu_7TeV_mcatnlo']=genvalid('WminusToENu_7TeV_mcatnlo_cff',step1GenDefaults,'w',666)
step1['WminusToMuNu_7TeV_mcatnlo']=genvalid('WminusToMuNu_7TeV_mcatnlo_cff',step1GenDefaults,'w',668)
step1['WplusToENu_7TeV_mcatnlo']=genvalid('WplusToENu_7TeV_mcatnlo_cff',step1GenDefaults,'w',665)
step1['WplusToMuNu_7TeV_mcatnlo']=genvalid('WplusToMuNu_7TeV_mcatnlo_cff',step1GenDefaults,'w',667)
step1['QCD_Ht-100To250_TuneD6T_7TeV_madgraph-tauola']=genvalid('Hadronizer_MgmMatchTuneD6T_7TeV_madgraph_tauola_cff',step1GenDefaults,'qcd',546)
step1['QCD_Ht-250To500_TuneD6T_7TeV_madgraph-tauola']=genvalid('Hadronizer_MgmMatchTuneD6T_7TeV_madgraph_tauola_cff',step1GenDefaults,'qcd',592)
step1['QCD_Ht-500To1000_TuneD6T_7TeV_madgraph-tauola']=genvalid('Hadronizer_MgmMatchTuneD6T_7TeV_madgraph_tauola_cff',step1GenDefaults,'qcd',594)
step1['TTJets_TuneD6T_7TeV_madgraph-tauola']=genvalid('Hadronizer_MgmMatchTuneD6T_7TeV_madgraph_tauola_cff',step1GenDefaults,'',846)
step1['WJetsLNu_TuneD6T_7TeV_madgraph-tauola']=genvalid('Hadronizer_MgmMatchTuneD6T_7TeV_madgraph_tauola_cff',step1GenDefaults,'w',882)
step1['ZJetsLNu_TuneD6T_7TeV_madgraph-tauola']=genvalid('Hadronizer_MgmMatchTuneD6T_7TeV_madgraph_tauola_cff',step1GenDefaults,'dy',851)
step1['QCD2Jets_Pt-40To120_TuneZ2_7TeV_alpgen_tauola']=genvalid('Hadronizer_Et20ExclTuneZ2_7TeV_alpgen_tauola_cff',step1GenDefaults,'qcd',785)
step1['QCD3Jets_Pt-40To120_TuneZ2_7TeV_alpgen_tauola']=genvalid('Hadronizer_Et20ExclTuneZ2_7TeV_alpgen_tauola_cff',step1GenDefaults,'qcd',786)
step1['QCD4Jets_Pt-40To120_TuneZ2_7TeV_alpgen_tauola']=genvalid('Hadronizer_Et20ExclTuneZ2_7TeV_alpgen_tauola_cff',step1GenDefaults,'qcd',787)
step1['QCD5Jets_Pt-40To120_TuneZ2_7TeV_alpgen_tauola']=genvalid('Hadronizer_Et20ExclTuneZ2_7TeV_alpgen_tauola_cff',step1GenDefaults,'qcd',832)
step1['TT0Jets_Et-40_TuneZ2_7TeV_alpgen_tauola']=genvalid('Hadronizer_Et48ExclTuneZ2_7TeV_alpgen_tauola_cff',step1GenDefaults,'',472)
step1['TT1Jets_Et-40_TuneZ2_7TeV_alpgen_tauola']=genvalid('Hadronizer_Et48ExclTuneZ2_7TeV_alpgen_tauola_cff',step1GenDefaults,'',475)
step1['TT2Jets_Et-40_TuneZ2_7TeV_alpgen_tauola']=genvalid('Hadronizer_Et48ExclTuneZ2_7TeV_alpgen_tauola_cff',step1GenDefaults,'',478)
step1['TT3Jets_Et-40_TuneZ2_7TeV_alpgen_tauola']=genvalid('Hadronizer_Et48ExclTuneZ2_7TeV_alpgen_tauola_cff',step1GenDefaults,'',481)
step1['W0Jets_Pt-0To100_TuneZ2_7TeV_alpgen_tauola']=genvalid('Hadronizer_Et20ExclTuneZ2_7TeV_alpgen_tauola_cff',step1GenDefaults,'w',397)
step1['W1Jets_Pt-0To100_TuneZ2_7TeV_alpgen_tauola']=genvalid('Hadronizer_Et20ExclTuneZ2_7TeV_alpgen_tauola_cff',step1GenDefaults,'w',398)
step1['W2Jets_Pt-0To100_TuneZ2_7TeV_alpgen_tauola']=genvalid('Hadronizer_Et20ExclTuneZ2_7TeV_alpgen_tauola_cff',step1GenDefaults,'w',399)
step1['W3Jets_Pt-0To100_TuneZ2_7TeV_alpgen_tauola']=genvalid('Hadronizer_Et20ExclTuneZ2_7TeV_alpgen_tauola_cff',step1GenDefaults,'w',400)
step1['Z0Jets_Pt-0To100_TuneZ2_7TeV_alpgen_tauola']=genvalid('Hadronizer_Et20ExclTuneZ2_7TeV_alpgen_tauola_cff',step1GenDefaults,'dy',440)
step1['Z1Jets_Pt-0To100_TuneZ2_7TeV_alpgen_tauola']=genvalid('Hadronizer_Et20ExclTuneZ2_7TeV_alpgen_tauola_cff',step1GenDefaults,'dy',441)
step1['Z2Jets_Pt-0To100_TuneZ2_7TeV_alpgen_tauola']=genvalid('Hadronizer_Et20ExclTuneZ2_7TeV_alpgen_tauola_cff',step1GenDefaults,'dy',442)
step1['Z3Jets-Pt_0To100_TuneZ2_7TeV_alpgen_tauola']=genvalid('Hadronizer_Et20ExclTuneZ2_7TeV_alpgen_tauola_cff',step1GenDefaults,'dy',443)

#PU1={'--pileup':'E7TeV_FlatDist10_2011EarlyData_inTimeOnly'}
PU1={'--pileup':'E7TeV_FlatDist10_2011EarlyData_50ns_PoissonOOT','--pileup_input':'dbs:/RelValProdMinBias/CMSSW_4_4_2_patch10-START44_V7_special_120119-v1/GEN-SIM-RAW'}
step1['ZmumuJets_Pt_20_300PU1']=merge([gen('ZmumuJets_Pt_20_300_GEN_7TeV_cfg',K250by100),PU1])
step1['TTbarPU2']=merge([step1['TTbar'],PU1])

step1['TTbarFSPU']=merge([{'--pileup':'FlatDist10_2011EarlyData_50ns'},step1['TTbarFS']])
##########################



# step2 
step2Defaults = { 'cfg'           : 'step2',
                  '-s'            : 'DIGI,L1,DIGI2RAW,HLT,RAW2DIGI,L1Reco',
                  '--datatier'    : 'GEN-SIM-DIGI-RAW-HLTDEBUG',
                  '--eventcontent': 'FEVTDEBUGHLT',
                  '--conditions'  : 'auto:startup',
                  }

step2 = {}

step2['DIGIPROD1']=merge([{'--eventcontent':'RAWSIM','--datatier':'GEN-SIM-RAW'},step2Defaults])
step2['DIGI']=merge([step2Defaults])
#step2['DIGI2']=merge([stCond,step2Defaults])
step2['DIGICOS']=merge([{'--scenario':'cosmics','--eventcontent':'FEVTDEBUG','--datatier':'GEN-SIM-DIGI-RAW'},stCond,step2Defaults])

step2['DIGIPU1']=merge([step2['DIGI'],PU1])

step2['DIGIHI']=merge([{'--inputCommands':'"keep *","drop *_simEcalPreshowerDigis_*_*"','-n':10},hiDefaults,step2Defaults])

#add this line when testing from an input file that is not strictly GEN-SIM
#addForAll(step2,{'--process':'DIGI'})

dataReco={'--conditions':'auto:com10',
          '-s':'RAW2DIGI,L1Reco,RECO,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias,DQM',
          '--datatier':'RECO,DQMROOT',
          '--eventcontent':'RECO,DQMROOT',
          '--data':'',
          '--magField':'AutoFromDBCurrent',
          '--customise':'Configuration/DataProcessing/RecoTLR.customisePPData',
          '--process':'reRECO',
          '--scenario':'pp',
          }
step2['RECOD']=merge([{'--scenario':'pp',},dataReco])
step2['RECOSKIMALCA']=merge([{'--inputCommands':'"keep *","drop *_*_*_RECO"'
                              },step2['RECOD']])
step2['RECOSKIM']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,DQM',
                          },step2['RECOSKIMALCA']])

step2['REPACKHID']=merge([{'--scenario':'HeavyIons',
                         '-s':'RAW2DIGI,REPACK',
                         '--datatier':'RAW',
                         '--eventcontent':'REPACKRAW'},
                        step2['RECOD']])
step2['REPACKHID'].pop('--customise')
step2['RECOHID10']=merge([{'--scenario':'HeavyIons',
                         '-s':'RAW2DIGI,L1Reco,RECO,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBiasHI+HcalCalMinBias+DtCalibHI,DQM',
                         '--customise':'Configuration/DataProcessing/RecoTLR.customiseCommonHI',
                         '--datatier':'RECO,DQMROOT',
                         '--eventcontent':'RECO,DQMROOT'},
                        step2['RECOD']])
step2['RECOHID11']=merge([{'--repacked':''},
                          step2['RECOHID10']])
step2['RECOHID10']['-s']+=',REPACK'
step2['RECOHID10']['--datatier']+=',RAW'
step2['RECOHID10']['--eventcontent']+=',REPACKRAW'

step2['TIER0']=merge([{'--customise':'Configuration/DataProcessing/RecoTLR.customisePrompt',
                       '-s':'RAW2DIGI,L1Reco,RECO,ALCAPRODUCER:@AllForPrompt,L1HwVal,DQM,ENDJOB',
                       '--datatier':'RECO,AOD,ALCARECO,DQMROOT',
                       '--eventcontent':'RECO,AOD,ALCARECO,DQMROOT',
                       '--process':'RECO'
                       },dataReco])
#step2['TIER0'].pop('--inputCommands')
step2['TIER0EXP']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCAPRODUCER:@StreamExpress,L1HwVal,DQM,ENDJOB',
                          '--datatier':'ALCARECO,DQM',
                          '--eventcontent':'ALCARECO,DQM',
                          '--customise':'Configuration/DataProcessing/RecoTLR.customiseExpress',
                          },step2['TIER0']])

step2['RECOCOSD']=merge([{'--scenario':'cosmics',
                          '-s':'RAW2DIGI,L1Reco,RECO,L1HwVal,DQM,ALCA:MuAlCalIsolatedMu+DtCalib',
                          '--customise':'Configuration/DataProcessing/RecoTLR.customiseCosmicData'
                          },dataReco])

step2HImixDefaults=merge([{'-n':'10',
                           '--himix':'',
                           '--filein':'file.root',
                           '--process':'HISIGNAL'
                           },hiDefaults,step1Defaults])
step2['Pyquen_GammaJet_pt20_2760GeV']=merge([{'cfg':'Pyquen_GammaJet_pt20_2760GeV_cfi'},step2HImixDefaults])
step2['Pyquen_DiJet_pt80to120_2760GeV']=merge([{'cfg':'Pyquen_DiJet_pt80to120_2760GeV_cfi'},step2HImixDefaults])
step2['Pyquen_ZeemumuJets_pt10_2760GeV']=merge([{'cfg':'Pyquen_ZeemumuJets_pt10_2760GeV_cfi'},step2HImixDefaults])

# step3 
step3Defaults = { 'cfg'           : 'step3',
                  '-s'            : 'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                  #'--filein'      : 'file:reco.root',
                  '--conditions'  : 'auto:startup',
                  '--no_exec'     : '',
                  '--datatier'    : 'GEN-SIM-RECO,DQM',
                  '--eventcontent': 'RECOSIM,DQM'
                  }

step3 = {}

step3['RECO']=merge([step3Defaults])
#step3['RECO2']=merge([stCond,step3Defaults])
step3['RECOPROD1']=merge([{ '-s' : 'RAW2DIGI,L1Reco,RECO', '--datatier' : 'GEN-SIM-RECO,AODSIM', '--eventcontent' : 'RECOSIM,AODSIM'},step3Defaults])
step3['RECOMU']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCA:@Mu','--datatier':'GEN-SIM-RECO','--eventcontent':'RECOSIM'},stCond,step3Defaults])
step3['RECOCOS']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCA:MuAlCalIsolatedMu,DQM','--datatier':'GEN-SIM-RECO','--eventcontent':'RECOSIM','--scenario':'cosmics'},stCond,step3Defaults])
step3['RECOMIN']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCA:SiStripCalZeroBias+SiStripCalMinBias+EcalCalPhiSym+EcalCalPi0Calib+EcalCalEtaCalib,VALIDATION,DQM'},stCond,step3Defaults])
step3['RECOQCD']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCA:@QCD,VALIDATION,DQM'},stCond,step3Defaults])

step3['RECOPU1']=merge([step3['RECO'],PU1])

step3['RECOHI']=merge([hiDefaults,step3Defaults])
step3['DIGIHISt3']=step2['DIGIHI']

step3['RECOHID11St3']=merge([{'cfg':'step3',
                              '--process':'ZStoRECO'},
                             step2['RECOHID11']])
step3['RECOHIR10D11']=merge([{'--filein':'file:step2_inREPACKRAW.root',
                              '--filtername':'reRECO'},
                             step3['RECOHID11St3']])

#add this line when testing from an input file that is not strictly GEN-SIM
#addForAll(step3,{'--hltProcess':'DIGI'})

step3['ALCACOSD']={'--conditions':'auto:com10',
                   '--datatier':'ALCARECO',
                   '--eventcontent':'ALCARECO',
                   '--scenario':'cosmics',
                   '-s':'ALCA:TkAlCosmics0T+MuAlGlobalCosmics+HcalCalHOCosmics+DQM'
                   }
step3['ALCAPROMPT']={'-s':'ALCA:PromptCalibProd',
                     '--filein':'file:TkAlMinBias.root',
                     '--conditions':'auto:com10',
                     '--datatier':'ALCARECO',
                     '--eventcontent':'ALCARECO'}

step3['HARVESTD']={'-s':'HARVESTING:dqmHarvesting',
                   '--conditions':'auto:com10',
                   '--filein':'file:step2_inDQM.root',
                   '--filetype':'DQM',
                   '--data':'',
                   '--scenario':'pp'}

# step4
step4Defaults = { 'cfg'           : 'step4',
                  '-s'            : 'ALCA:TkAlMuonIsolated+TkAlMinBias+EcalCalElectron+HcalCalIsoTrk+MuAlOverlaps',
                  '-n'            : 1000,
                  #'--filein'      : 'file:reco.root',
                  '--conditions'  : 'auto:startup',
                  '--datatier'    : 'ALCARECO',
                  '--eventcontent': 'ALCARECO',
                  }
step4 = {}

step4['ALCATT1']=merge([step4Defaults])
step4['ALCATT2']=merge([stCond,step4Defaults])
step4['ALCAMIN']=merge([{'-s':'ALCA:TkAlMinBias'},stCond,step4Defaults])
#step4['ALCAQCD']=merge([{'-s':'ALCA:HcalCalIsoTrk+HcalCalDijets'},stCond,step4Defaults])
#step4['ALCAMU']=merge([{'-s':'ALCA:@Mu'},stCond,step4Defaults])
step4['ALCACOS']=merge([{'-s':'ALCA:TkAlCosmics0T+MuAlGlobalCosmics+HcalCalHOCosmics'},stCond,step4Defaults])
step4['ALCABH']=merge([{'-s':'ALCA:TkAlBeamHalo+MuAlBeamHaloOverlaps+MuAlBeamHalo'},stCond,step4Defaults])
step4['ALCAELE']=merge([{'-s':'ALCA:EcalCalElectron'},stCond,step4Defaults])

step4['ALCAHARVD']={'-s':'ALCAHARVEST:BeamSpotByRun+BeamSpotByLumi',
                    '--conditions':'auto:com10',
                    '--scenario':'pp',
                    '--data':'',
                    '--filein':'file:PromptCalibProd.root'}

step4['RECOHISt4']=step3['RECOHI']

step3['ALCANZS']=merge([{'-s':'ALCA:HcalCalMinBias','cfg':'step3','--mc':''},step4Defaults])
step2['HARVGEN']={'-s':'HARVESTING:genHarvesting',
                  '--harvesting':'AtJobEnd',
                  '--conditions':'auto:startup',
                  '--mc':'',
                  '--filein':'file:step1.root'
                  }

step4['HARVEST']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                   '--conditions':'auto:startup',
                   '--mc':'',
                   '--scenario':'pp'}
step4['ALCASPLIT']={'-s':'ALCAOUTPUT:@AllForPrompt',
                    '--conditions':'auto:com10',
                    '--scenario':'pp',
                    '--data':'',
                    '--triggerResultsProcess':'RECO',
                    '--filein':'file:step2_inALCARECO.root'}

step4['SKIMD']={'-s':'SKIM:all',
                '--conditions':'auto:com10',
                '--data':'',
                '--scenario':'pp',
                '--filein':'file:step2.root',
                '--secondfilein':'filelist:step1_dbsquery.log'}


step4['SKIMCOSD']={'-s':'SKIM:all',
                   '--conditions':'auto:com10',
                   '--data':'',
                   '--scenario':'cosmics',
                   '--filein':'file:step2.root',
                   '--secondfilein':'filelist:step1_dbsquery.log'}
                 

#### for special wfs ###
#step1['TTbar_REDIGI_RERECO']=merge([{'cfg':'TTbar_Tauola_7TeV_cfi',
#                                     '-s':'GEN,SIM,DIGI,L1,DIGI2RAW,HLT:GRun,RAW2DIGI,L1Reco,RECO,ALCA:MuAlCalIsolatedMu+DtCalib,VALIDATION,DQM',
#                                     '--datatier':'GEN-SIM-DIGI-RAW-HLTDEBUG-RECO,DQM',
#                                     '--eventcontent':'FEVTDEBUGHLT,DQM'},
#                                    K9by50,stCond,step1Defaults])
#step2['DIGI2RECO']=merge([{'-s':'DIGI,L1,DIGI2RAW,HLT:GRun,RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
#                           '--filtername':'DIGItoRECO',
#                           '--process':'RECO',
#                           '--eventcontent':'RECOSIM,DQM',
#                           '--datatier':'GEN-SIM-RECO,DQM',
#                           'cfg':'step2'},
#                            stCond,step3Defaults])
step4['RECOFROMRECO']=merge([{'-s':'RECO',
                              '--filtername':'RECOfromRECO',
                              '--process':'reRECO',
                              '--datatier':'AODSIM',
                              '--eventcontent':'AODSIM',
                              'cfg':'step4'},
                             stCond,step3Defaults])
step2['RECOFROMRECOSt2']=merge([{'cfg':'step2'},step4['RECOFROMRECO']])
step3['RECODFROMRAWRECO']=merge([{'-s':'RAW2DIGI:RawToDigi_noTk,L1Reco,RECO:reconstruction_noTracking',
                                  '--filtername':'RECOfromRAWRECO',
                                  '--process':'rereRECO',
                                  '--datatier':'AOD',
                                  '--eventcontent':'AOD',
                                  '--secondfilein':'filelist:step1_dbsquery.log',
                                  'cfg':'step3'},
                                 step2['RECOD']])


# to handle things easier in other places, make a list of all the steps:
stepList = [step1, step2, step3, step4]


