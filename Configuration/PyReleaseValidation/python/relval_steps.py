

class Matrix(dict):
    def __setitem__(self,key,value):
        if key in self:
            print "ERROR in Matrix"
            print "overwritting",key,"not allowed"
        else:
            self.update({float(key):WF(float(key),value)})

            
#the class to collect all possible steps
class Steps(dict):
    def __setitem__(self,key,value):
        if key in self:
            print "ERROR in Step"
            print "overwritting",key,"not allowed"
            import sys
            sys.exit(-9)
        else:
            self.update({key:value})
            # make the python file named <step>.py
            #if not '--python' in value:                self[key].update({'--python':'%s.py'%(key,)})

    def overwrite(self,keypair):
        value=self[keypair[1]]
        print "overwritting step",keypair[0],"with",keypair[1],str(value)
        self.update({keypair[0]:value})
        
class WF(list):
    def __init__(self,n,l):
        self.extend(l)
        self.num=n
        #the actual steps of this WF
        self.steps=[]

        
    def interpret(self,stepsDict):
        for s in self:
            steps.append(stepsDict[s])
    
InputInfoNDefault=2000000    
class InputInfo(object):
    def __init__(self,dataSet,label='',run=[],files=1000,events=InputInfoNDefault,split=10,location='CAF') :
        self.run = run
        self.files = files
        self.events = events
        self.location = location
        self.label = label
        self.dataSet = dataSet
        self.split=split
        
    def dbs(self):
        command='dbs search --noheader --query "find file where dataset like '+self.dataSet
        def requ(r):
            return 'run=%d'%(r,)
        if len(self.run)!=0:
            command+=' and ('+' or '.join(map(requ,self.run))+' )'
        command+='"'
        return command
    def __str__(self):
        return 'input from: %s with run: %s'%(self.dataSet,str(self.run))
    
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

steps = Steps()
wmsplit = {}

#### Production test section ####
steps['ProdMinBias']=merge([{'cfg':'MinBias_8TeV_cfi','--relval':'9000,300'},step1Defaults])
steps['ProdTTbar']=merge([{'cfg':'TTbar_Tauola_8TeV_cfi','--relval':'9000,100'},step1Defaults])
steps['ProdQCD_Pt_3000_3500']=merge([{'cfg':'QCD_Pt_3000_3500_8TeV_cfi','--relval':'9000,50'},step1Defaults])

#### data ####
#list of run to harvest for 2010A: 144086,144085,144084,144083,144011,139790,139789,139788,139787,138937,138934,138924,138923
#list of run to harvest for 2010B: 149442,149291,149181,149011,148822,147929,147115,146644
Run2010ASk=[138937,138934,138924,138923,139790,139789,139788,139787,144086,144085,144084,144083,144011]
Run2010BSk=[146644,147115,147929,148822,149011,149181,149182,149291,149294,149442]
steps['MinimumBias2010A']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2010A-valskim-v6/RAW-RECO',label='run2010A',location='STD',run=Run2010ASk)}
steps['MinimumBias2010B']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2010B-valskim-v2/RAW-RECO',label='run2010B',run=Run2010BSk)}
steps['WZMuSkim2010A']={'INPUT':InputInfo(dataSet='/Mu/Run2010A-WZMu-Nov4Skim_v1/RAW-RECO',label='wzMu2010A',run=Run2010ASk)}
steps['WZMuSkim2010B']={'INPUT':InputInfo(dataSet='/Mu/Run2010B-WZMu-Nov4Skim_v1/RAW-RECO',label='wzMu2010B',run=Run2010BSk)}
steps['WZEGSkim2010A']={'INPUT':InputInfo(dataSet='/EG/Run2010A-WZEG-Nov4Skim_v1/RAW-RECO',label='wzEG2010A',run=Run2010ASk)}
steps['WZEGSkim2010B']={'INPUT':InputInfo(dataSet='/Electron/Run2010B-WZEG-Nov4Skim_v1/RAW-RECO',label='wzEG2010B',run=Run2010BSk)}

steps['RunCosmicsA']={'INPUT':InputInfo(dataSet='/Cosmics/Run2010A-v1/RAW',label='cos2010A',run=[142089],events=100000)}
Run2010B=[149011]
steps['RunMinBias2010B']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2010B-RelValRawSkim-v1/RAW',label='mb2010B',run=Run2010B,events=100000)}
steps['RunMu2010B']={'INPUT':InputInfo(dataSet='/Mu/Run2010B-RelValRawSkim-v1/RAW',label='mu2010B',run=Run2010B,events=100000)}
steps['RunElectron2010B']={'INPUT':InputInfo(dataSet='/Electron/Run2010B-RelValRawSkim-v1/RAW',label='electron2010B',run=Run2010B,events=100000)}
steps['RunPhoton2010B']={'INPUT':InputInfo(dataSet='/Photon/Run2010B-RelValRawSkim-v1/RAW',label='photon2010B',run=Run2010B,events=100000)}
steps['RunJet2010B']={'INPUT':InputInfo(dataSet='/Jet/Run2010B-RelValRawSkim-v1/RAW',label='jet2010B',run=Run2010B,events=100000)}

#list of run to harvest 2011A: 165121, 172802,
Run2011ASk=[165121,172802]
steps['ValSkim2011A']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2011A-ValSkim-PromptSkim-v6/RAW-RECO',label='run2011A',location='STD',run=Run2011ASk)}
steps['WMuSkim2011A']={'INPUT':InputInfo(dataSet='/SingleMu/Run2011A-WMu-PromptSkim-v6/RAW-RECO',label='wMu2011A',location='STD',run=Run2011ASk)}
steps['WElSkim2011A']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2011A-WElectron-PromptSkim-v6/RAW-RECO',label='wEl2011A',location='STD',run=Run2011ASk)}
steps['ZMuSkim2011A']={'INPUT':InputInfo(dataSet='/DoubleMu/Run2011A-ZMu-PromptSkim-v6/RAW-RECO',label='zMu2011A',location='STD',run=Run2011ASk)}
steps['ZElSkim2011A']={'INPUT':InputInfo(dataSet='/DoubleElectron/Run2011A-ZElectron-PromptSkim-v6/RAW-RECO',label='zEl2011A',location='STD',run=Run2011ASk)}
steps['HighMet2011A']={'INPUT':InputInfo(dataSet='/Jet/Run2011A-HighMET-PromptSkim-v6/RAW-RECO',label='hMet2011A',location='STD',run=Run2011ASk)}

steps['RunCosmics2011A']={'INPUT':InputInfo(dataSet='/Cosmics/Run2011A-v1/RAW',label='cos2011A',run=[160960],events=100000,location='STD')}
Run2011A=[165121]
steps['RunMinBias2011A']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2011A-v1/RAW',label='mb2011A',run=Run2011A,events=100000,location='STD')}
steps['RunMu2011A']={'INPUT':InputInfo(dataSet='/SingleMu/Run2011A-v1/RAW',label='mu2011A',run=Run2011A,events=100000)}
steps['RunElectron2011A']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2011A-v1/RAW',label='electron2011A',run=Run2011A,events=100000)}
steps['RunPhoton2011A']={'INPUT':InputInfo(dataSet='/Photon/Run2011A-v1/RAW',label='photon2011A',run=Run2011A,events=100000)}
steps['RunJet2011A']={'INPUT':InputInfo(dataSet='/Jet/Run2011A-v1/RAW',label='jet2011A',run=Run2011A,events=100000)}

Run2011B=[177719]
Run2011BSk=[177719,177790,177096,175874]
steps['RunMinBias2011B']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2011B-v1/RAW',label='mb2011B',run=Run2011B,events=100000,location='STD')}
steps['RunMu2011B']={'INPUT':InputInfo(dataSet='/SingleMu/Run2011B-v1/RAW',label='mu2011B',run=Run2011B,events=100000)}
steps['RunElectron2011B']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2011B-v1/RAW',label='electron2011B',run=Run2011B,events=100000)}
steps['RunPhoton2011B']={'INPUT':InputInfo(dataSet='/Photon/Run2011B-v1/RAW',label='photon2011B',run=Run2011B,events=100000)}
steps['RunJet2011B']={'INPUT':InputInfo(dataSet='/Jet/Run2011B-v1/RAW',label='jet2011B',run=Run2011B,events=100000)}

steps['ValSkim2011B']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2011B-ValSkim-PromptSkim-v1/RAW-RECO',label='run2011B',location='STD',run=Run2011BSk)}
steps['WMuSkim2011B']={'INPUT':InputInfo(dataSet='/SingleMu/Run2011B-WMu-PromptSkim-v1/RAW-RECO',label='wMu2011B',location='STD',run=Run2011BSk)}
steps['WElSkim2011B']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2011B-WElectron-PromptSkim-v1/RAW-RECO',label='wEl2011B',location='STD',run=Run2011BSk)}
steps['ZMuSkim2011B']={'INPUT':InputInfo(dataSet='/DoubleMu/Run2011B-ZMu-PromptSkim-v1/RAW-RECO',label='zMu2011B',location='STD',run=Run2011BSk)}
steps['ZElSkim2011B']={'INPUT':InputInfo(dataSet='/DoubleElectron/Run2011B-ZElectron-PromptSkim-v1/RAW-RECO',label='zEl2011B',run=Run2011BSk)}
steps['HighMet2011B']={'INPUT':InputInfo(dataSet='/Jet/Run2011B-HighMET-PromptSkim-v1/RAW-RECO',label='hMet2011B',run=Run2011BSk)}

steps['RunHI2010']={'INPUT':InputInfo(dataSet='/HIAllPhysics/HIRun2010-v1/RAW',label='hi2010',run=[152698],events=10000,location='STD')}
steps['RunHI2011']={'INPUT':InputInfo(dataSet='/HIAllPhysics/HIRun2011A-v1/RAW',label='hi2011',run=[174773],events=10000,location='STD')}


Run2012A=[191226]
steps['RunMinBias2012A']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2012A-v1/RAW',label='mb2012A',run=Run2012A, events=100000,location='STD')}
steps['RunTau2012A']={'INPUT':InputInfo(dataSet='/Tau/Run2012A-v1/RAW',label='tau2012A', run=Run2012A, events=100000,location='STD')}
steps['RunMET2012A']={'INPUT':InputInfo(dataSet='/MET/Run2012A-v1/RAW',label='met2012A', run=Run2012A, events=100000,location='STD')}
steps['RunMu2012A']={'INPUT':InputInfo(dataSet='/SingleMu/Run2012A-v1/RAW',label='mu2012A', run=Run2012A, events=100000,location='STD')}
steps['RunElectron2012A']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2012A-v1/RAW',label='electron2012A', run=Run2012A, events=100000,location='STD')}
steps['RunJet2012A']={'INPUT':InputInfo(dataSet='/Jet/Run2012A-v1/RAW',label='jet2012A', run=Run2012A, events=100000,location='STD')}


Run2012B=[194533]
Run2012Bsk=Run2012B+[194912,195016]
steps['RunMinBias2012B']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2012B-v1/RAW',label='mb2012B',run=Run2012B, events=100000,location='STD')}
steps['RunMu2012B']={'INPUT':InputInfo(dataSet='/SingleMu/Run2012B-v1/RAW',label='mu2012B',location='STD',run=Run2012B)}
steps['RunPhoton2012B']={'INPUT':InputInfo(dataSet='/SinglePhoton/Run2012B-v1/RAW',label='photon2012B',location='STD',run=Run2012B)}
steps['RunEl2012B']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2012B-v1/RAW',label='electron2012B',location='STD',run=Run2012B)}
steps['RunJet2012B']={'INPUT':InputInfo(dataSet='/JetHT/Run2012B-v1/RAW',label='jet2012B',location='STD',run=Run2012B)}
steps['ZMuSkim2012B']={'INPUT':InputInfo(dataSet='/SingleMu/Run2012B-ZMu-PromptSkim-v1/RAW-RECO',label='zMu2012B',location='CAF',run=Run2012Bsk)}
steps['WElSkim2012B']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2012B-WElectron-PromptSkim-v1/USER',label='wEl2012B',location='STD',run=Run2012Bsk)}
steps['ZElSkim2012B']={'INPUT':InputInfo(dataSet='/DoubleElectron/Run2012B-ZElectron-PromptSkim-v1/RAW-RECO',label='zEl2012B',location='STD',run=Run2012Bsk)}

Run2012C=[199812]
Run2012Csk=Run2012C+[]
steps['RunMinBias2012C']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2012C-v1/RAW',label='mb2012C',run=Run2012C, events=100000,location='STD')}
steps['RunMu2012C']={'INPUT':InputInfo(dataSet='/SingleMu/Run2012C-v1/RAW',label='mu2012C',location='STD',run=Run2012C)}
steps['RunPhoton2012C']={'INPUT':InputInfo(dataSet='/SinglePhoton/Run2012C-v1/RAW',label='photon2012C',location='STD',run=Run2012C)}
steps['RunEl2012C']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2012C-v1/RAW',label='electron2012C',location='STD',run=Run2012C)}
steps['RunJet2012C']={'INPUT':InputInfo(dataSet='/JetHT/Run2012C-v1/RAW',label='jet2012C',location='STD',run=Run2012C)}
steps['ZMuSkim2012C']={'INPUT':InputInfo(dataSet='/SingleMu/Run2012C-ZMu-PromptSkim-v3/RAW-RECO',label='zMu2012C',location='CAF',run=Run2012Csk)}
steps['WElSkim2012C']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2012C-WElectron-PromptSkim-v3/USER',label='wEl2012C',location='STD',run=Run2012Csk)}
steps['ZElSkim2012C']={'INPUT':InputInfo(dataSet='/DoubleElectron/Run2012C-ZElectron-PromptSkim-v3/RAW-RECO',label='zEl2012C',location='STD',run=Run2012Csk)}

#### Standard release validation samples ####

stCond={'--conditions':'auto:startup'}
def Kby(N,s):
    return {'--relval':'%s000,%s'%(N,s)}


def gen(fragment,howMuch):
    global step1Defaults
    return merge([{'cfg':fragment},howMuch,step1Defaults])

steps['MinBias']=gen('MinBias_8TeV_cfi',Kby(9,300))
steps['QCD_Pt_3000_3500']=gen('QCD_Pt_3000_3500_8TeV_cfi',Kby(9,25))
steps['QCD_Pt_600_800']=gen('QCD_Pt_600_800_8TeV_cfi',Kby(9,50))
steps['QCD_Pt_80_120']=gen('QCD_Pt_80_120_8TeV_cfi',Kby(9,100))
steps['SingleElectronPt10']=gen('SingleElectronPt10_cfi',Kby(9,3000))
steps['SingleElectronPt35']=gen('SingleElectronPt35_cfi',Kby(9,500))
steps['SingleElectronPt1000']=gen('SingleElectronPt1000_cfi',Kby(9,50))
steps['SingleGammaPt10']=gen('SingleGammaPt10_cfi',Kby(9,3000))
steps['SingleGammaPt35']=gen('SingleGammaPt35_cfi',Kby(9,500))
steps['SingleMuPt1']=gen('SingleMuPt1_cfi',Kby(25,1000))
steps['SingleMuPt10']=gen('SingleMuPt10_cfi',Kby(25,500))
steps['SingleMuPt100']=gen('SingleMuPt100_cfi',Kby(9,500))
steps['SingleMuPt1000']=gen('SingleMuPt1000_cfi',Kby(9,500))
steps['TTbar']=gen('TTbar_Tauola_8TeV_cfi',Kby(9,100))
steps['TTbarLepton']=gen('TTbarLepton_Tauola_8TeV_cfi',Kby(9,100))
steps['ZEE']=gen('ZEE_8TeV_cfi',Kby(9,100))
steps['Wjet_Pt_80_120']=gen('Wjet_Pt_80_120_8TeV_cfi',Kby(9,100))
steps['Wjet_Pt_3000_3500']=gen('Wjet_Pt_3000_3500_8TeV_cfi',Kby(9,50))
steps['LM1_sfts']=gen('LM1_sfts_8TeV_cfi',Kby(9,100))
steps['QCD_FlatPt_15_3000']=gen('QCDForPF_8TeV_cfi',Kby(9,100))
steps['QCD_FlatPt_15_3000HS']=gen('QCDForPF_8TeV_cfi',Kby(50,100))

def identitySim(wf):
    return merge([{'--restoreRND':'SIM','--process':'SIM2'},wf])

steps['SingleMuPt10_ID']=identitySim(steps['SingleMuPt10'])
steps['TTbar_ID']=identitySim(steps['TTbar'])

baseDataSetRelease=[
    'CMSSW_6_1_0_pre6-START61_V5-v1',#'CMSSW_6_0_0-START60_V4-v1',
    'CMSSW_6_1_0_pre6-STARTHI61_V6-v1',#'CMSSW_6_0_0-STARTHI60_V4-v1',
    'CMSSW_6_1_0_pre6-START61_V5-v2',#'CMSSW_6_0_0-PU_START60_V4-v1',
    'CMSSW_6_1_0_pre6-START61_V5_FastSim-v1'#'CMSSW_6_0_0-START60_V4_FastSim-v1'
    ]

steps['MinBiasINPUT']={'INPUT':InputInfo(dataSet='/RelValMinBias/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['QCD_Pt_3000_3500INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_3000_3500/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['QCD_Pt_600_800INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_600_800/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['QCD_Pt_80_120INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_80_120/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['SingleElectronPt10INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleElectronPt10/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['SingleElectronPt1000INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleElectronPt1000/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['SingleElectronPt35INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleElectronPt35/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['SingleGammaPt10INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleGammaPt10/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['SingleGammaPt35INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleGammaPt35/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['SingleMuPt1INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt1/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['SingleMuPt10INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt10/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['SingleMuPt10IdINPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt10/%s/GEN-SIM-DIGI-RAW-HLTDEBUG'%(baseDataSetRelease[0],),location='STD')}
steps['SingleMuPt10FSIdINPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt10/%s/GEN-SIM-DIGI-RECO'%(baseDataSetRelease[3],),location='STD')}
steps['SingleMuPt100INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt100/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['SingleMuPt1000INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt1000/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['TTbarINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['TTbarIdINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/%s/GEN-SIM-DIGI-RAW-HLTDEBUG'%(baseDataSetRelease[0],),location='STD')}
steps['TTbarFSIdINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/%s/GEN-SIM-DIGI-RECO'%(baseDataSetRelease[0],),location='STD')}
steps['TTbarLeptonINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbarLepton/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['OldTTbarINPUT']={'INPUT':InputInfo(dataSet='/RelValProdTTbar/CMSSW_5_0_0_pre6-START50_V5-v1/GEN-SIM-RECO',location='STD')}
steps['OldGenSimINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/CMSSW_4_4_2-START44_V7-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',location='STD')}
steps['Wjet_Pt_80_120INPUT']={'INPUT':InputInfo(dataSet='/RelValWjet_Pt_80_120/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['Wjet_Pt_3000_3500INPUT']={'INPUT':InputInfo(dataSet='/RelValWjet_Pt_3000_3500/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['LM1_sftsINPUT']={'INPUT':InputInfo(dataSet='/RelValLM1_sfts/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['QCD_FlatPt_15_3000INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_FlatPt_15_3000/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}

steps['QCD_FlatPt_15_3000HSINPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_FlatPt_15_3000/CMSSW_5_2_2-PU_START52_V4_special_120326-v1/GEN-SIM',location='STD')}
steps['QCD_FlatPt_15_3000HS__DIGIPU1INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_FlatPt_15_3000/CMSSW_5_2_2-PU_START52_V4_special_120326-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',location='STD')}
steps['TTbar__DIGIPU1INPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/CMSSW_5_2_2-PU_START52_V4_special_120326-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',location='STD')}

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

K25by250={'--relval':'25000,250'}
steps['SingleElectronE120EHCAL']=merge([{'cfg':'SingleElectronE120EHCAL_cfi'},ecalHcal,K25by250,step1Defaults])
steps['SinglePiE50HCAL']=merge([{'cfg':'SinglePiE50HCAL_cfi'},ecalHcal,K25by250,step1Defaults])

steps['MinBiasHS']=gen('MinBias_8TeV_cfi',Kby(25,300))
steps['InclusiveppMuX']=gen('InclusiveppMuX_8TeV_cfi',K110000by45000)
steps['SingleElectronFlatPt5To100']=gen('SingleElectronFlatPt5To100_cfi',K250by250)
steps['SinglePiPt1']=gen('SinglePiPt1_cfi',K250by250)
steps['SingleMuPt1HS']=gen('SingleMuPt1_cfi',Kby(250,1000))
steps['ZPrime5000Dijet']=gen('ZPrime5000JJ_8TeV_cfi',K250by100)
steps['SinglePi0E10']=gen('SinglePi0E10_cfi',K250by100)
steps['SinglePiPt10']=gen('SinglePiPt10_cfi',K250by250)
steps['SingleGammaFlatPt10To100']=gen('SingleGammaFlatPt10To100_cfi',K250by250)
steps['SingleTauPt50Pythia']=gen('SingleTaupt_50_cfi',K250by100)
steps['SinglePiPt100']=gen('SinglePiPt100_cfi',K250by250)


def genS(fragment,howMuch):
    global step1Defaults,stCond
    return merge([{'cfg':fragment},stCond,howMuch,step1Defaults])

steps['Higgs200ChargedTaus']=genS('H200ChargedTaus_Tauola_8TeV_cfi',Kby(9,100))
steps['JpsiMM']=genS('JpsiMM_8TeV_cfi',Kby(66,1000))
steps['WE']=genS('WE_8TeV_cfi',Kby(9,100))
steps['WM']=genS('WM_8TeV_cfi',Kby(9,200))
steps['WpM']=genS('WpM_8TeV_cfi',Kby(9,200))
steps['ZMM']=genS('ZMM_8TeV_cfi',Kby(18,300))
steps['ZpMM']=genS('ZpMM_8TeV_cfi',Kby(9,200))

steps['ZTT']=genS('ZTT_Tauola_All_hadronic_8TeV_cfi',Kby(9,150))
steps['H130GGgluonfusion']=genS('H130GGgluonfusion_8TeV_cfi',Kby(9,100))
steps['PhotonJets_Pt_10']=genS('PhotonJet_Pt_10_8TeV_cfi',Kby(9,150))
steps['QQH1352T_Tauola']=genS('QQH1352T_Tauola_8TeV_cfi',Kby(9,100))
steps['ZmumuJets_Pt_20_300']=gen('ZmumuJets_Pt_20_300_GEN_8TeV_cfg',Kby(250,100))
steps['ADDMonoJet_d3MD3']=genS('ADDMonoJet_8TeV_d3MD3_cfi',Kby(9,100))

steps['MinBias2INPUT']={'INPUT':InputInfo(dataSet='/RelValMinBias/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['Higgs200ChargedTausINPUT']={'INPUT':InputInfo(dataSet='/RelValHiggs200ChargedTaus/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['QCD_Pt_3000_3500_2INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_3000_3500/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['QCD_Pt_80_120_2INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_80_120/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['JpsiMMINPUT']={'INPUT':InputInfo(dataSet='/RelValJpsiMM/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['TTbar2INPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['WEINPUT']={'INPUT':InputInfo(dataSet='/RelValWE/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['WMINPUT']={'INPUT':InputInfo(dataSet='/RelValWM/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['ZEEINPUT']={'INPUT':InputInfo(dataSet='/RelValZEE/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['ZMMINPUT']={'INPUT':InputInfo(dataSet='/RelValZMM/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['ZTTINPUT']={'INPUT':InputInfo(dataSet='/RelValZTT/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['H130GGgluonfusionINPUT']={'INPUT':InputInfo(dataSet='/RelValH130GGgluonfusion/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['PhotonJets_Pt_10INPUT']={'INPUT':InputInfo(dataSet='/RelValPhotonJets_Pt_10/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['QQH1352T_TauolaINPUT']={'INPUT':InputInfo(dataSet='/RelValQQH1352T_Tauola/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['ADDMonoJet_d3MD3INPUT']={'INPUT':InputInfo(dataSet='/RelValADDMonoJet_d3MD3/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['WpMINPUT']={'INPUT':InputInfo(dataSet='/RelValZpMM/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['ZpMMINPUT']={'INPUT':InputInfo(dataSet='/RelValWpM/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}

steps['ZmumuJets_Pt_20_300INPUT']={'INPUT':InputInfo(dataSet='/RelValZmumuJets_Pt_20_300/%s/GEN-SIM'%(baseDataSetRelease[2],),location='STD')}
                                

steps['Cosmics']=merge([{'cfg':'UndergroundCosmicMu_cfi.py','--scenario':'cosmics'},Kby(666,100000),step1Defaults])
steps['BeamHalo']=merge([{'cfg':'BeamHalo_cfi.py','--scenario':'cosmics'},Kby(9,100),step1Defaults])

steps['CosmicsINPUT']={'INPUT':InputInfo(dataSet='/RelValCosmics/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['BeamHaloINPUT']={'INPUT':InputInfo(dataSet='/RelValBeamHalo/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}

steps['QCD_Pt_50_80']=genS('QCD_Pt_50_80_8TeV_cfi',K250by100)
steps['QCD_Pt_15_20']=genS('QCD_Pt_15_20_8TeV_cfi',K250by100)
steps['ZTTHS']=merge([K250by100,steps['ZTT']])
steps['QQH120Inv']=genS('QQH120Inv_8TeV_cfi',K250by100)
steps['TTbar2HS']=merge([K250by100,steps['TTbar']])
steps['JpsiMM_Pt_20_inf']=genS('JpsiMM_Pt_20_inf_8TeV_cfi',K700by280)
steps['QCD_Pt_120_170']=genS('QCD_Pt_120_170_8TeV_cfi',K250by100)
steps['H165WW2L']=genS('H165WW2L_Tauola_8TeV_cfi',K250by100)
steps['UpsMM']=genS('UpsMM_8TeV_cfi',K562by225)
steps['RSGrav']=genS('RS750_quarks_and_leptons_8TeV_cff',K250by100)
steps['QCD_Pt_80_120_2HS']=merge([K250by100,steps['QCD_Pt_80_120']])
steps['bJpsiX']=genS('bJpsiX_8TeV_cfi',K3250000by1300000)
steps['QCD_Pt_30_50']=genS('QCD_Pt_30_50_8TeV_cfi',K250by100)
steps['H200ZZ4L']=genS('H200ZZ4L_Tauola_8TeV_cfi',K250by100)
steps['LM9p']=genS('LM9p_8TeV_cff',K250by100)
steps['QCD_Pt_20_30']=genS('QCD_Pt_20_30_8TeV_cfi',K250by100)
steps['QCD_Pt_170_230']=genS('QCD_Pt_170_230_8TeV_cfi',K250by100)


## upgrade dedicated wf
step1Upgpixphase1Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'DESIGN60_V5::All',
                             '--beamspot' : 'Gauss',
                             '--datatier' : 'GEN-SIM',
                             '--eventcontent': 'FEVTDEBUG',
                             '--slhc' : 'Phase1_R30F12'
                             }

steps['FourMuPt1_200_UPGphase1']=merge([{'cfg':'FourMuPt_1_200_cfi','--relval':'10000,100'},step1Upgpixphase1Defaults])
steps['MinBias_UPGphase1_14']=merge([{'cfg':'MinBias_TuneZ2star_14TeV_pythia6_cff','--relval':'15000,250'},step1Upgpixphase1Defaults])
steps['TTbar_Tauola_UPGphase1_14']=merge([{'cfg':'TTbar_Tauola_14TeV_cfi','--relval':'10000,100'},step1Upgpixphase1Defaults])

## pPb tests
step1PPbDefaults={'--beamspot':'Realistic8TeVCollisionPPbBoost'}
steps['AMPT_PPb_5020GeV_MinimumBias']=merge([{'-n':10},step1PPbDefaults,genS('AMPT_PPb_5020GeV_MinimumBias_cfi',Kby(9,100))])

## heavy ions tests
U500by1={'--relval': '500,1'}
U80by1={'--relval': '80,1'}

hiDefaults={'--conditions':'auto:starthi_HIon',
           '--scenario':'HeavyIons'}

steps['HydjetQ_MinBias_2760GeV']=merge([{'-n':1},hiDefaults,genS('Hydjet_Quenched_MinBias_2760GeV_cfi',U500by1)])
steps['HydjetQ_MinBias_2760GeVINPUT']={'INPUT':InputInfo(dataSet='/RelValHydjetQ_MinBias_2760GeV/%s/GEN-SIM'%(baseDataSetRelease[1],),location='STD')}
steps['HydjetQ_B0_2760GeV']=merge([{'-n':1},hiDefaults,genS('Hydjet_Quenched_B0_2760GeV_cfi',U80by1)])
steps['HydjetQ_B0_2760GeVINPUT']={'INPUT':InputInfo(dataSet='/RelValHydjetQ_B0_2760GeV/%s/GEN-SIM'%(baseDataSetRelease[1],),location='STD')}
steps['HydjetQ_B8_2760GeV']=merge([{'-n':1},hiDefaults,genS('Hydjet_Quenched_B8_2760GeV_cfi',U80by1)])
steps['HydjetQ_B8_2760GeVINPUT']={'INPUT':InputInfo(dataSet='/RelValHydjetQ_B8_2760GeV/%s/GEN-SIM'%(baseDataSetRelease[1],),location='CAF')}



def changeRefRelease(steps,listOfPairs):
    for s in steps:
        if ('INPUT' in steps[s]):
            oldD=steps[s]['INPUT'].dataSet
            for (ref,newRef) in listOfPairs:
                if  ref in oldD:
                    steps[s]['INPUT'].dataSet=oldD.replace(ref,newRef)
        if '--pileup_input' in steps[s]:
            for (ref,newRef) in listOfPairs:
                if ref in steps[s]['--pileup_input']:
                    steps[s]['--pileup_input']=steps[s]['--pileup_input'].replace(ref,newRef)
        
def addForAll(steps,d):
    for s in steps:
        steps[s].update(d)



#### fastsim section ####
##no forseen to do things in two steps GEN-SIM then FASTIM->end: maybe later
step1FastDefaults =merge([{'-s':'GEN,FASTSIM,HLT:@relval,VALIDATION',
                           '--eventcontent':'FEVTDEBUGHLT,DQM',
                           '--datatier':'GEN-SIM-DIGI-RECO,DQM',
                           '--relval':'27000,3000'},
                          step1Defaults])
K100by500={'--relval':'100000,500'}
K100byK2={'--relval':'100000,2000'}
steps['TTbarFS']=merge([{'cfg':'TTbar_Tauola_8TeV_cfi'},Kby(100,1000),step1FastDefaults])
steps['SingleMuPt1FS']=merge([{'cfg':'SingleMuPt1_cfi'},step1FastDefaults])
steps['SingleMuPt10FS']=merge([{'cfg':'SingleMuPt10_cfi'},step1FastDefaults])
steps['SingleMuPt100FS']=merge([{'cfg':'SingleMuPt100_cfi'},step1FastDefaults])
steps['SinglePiPt1FS']=merge([{'cfg':'SinglePiPt1_cfi'},step1FastDefaults])
steps['SinglePiPt10FS']=merge([{'cfg':'SinglePiPt10_cfi'},step1FastDefaults])
steps['SinglePiPt100FS']=merge([{'cfg':'SinglePiPt100_cfi'},step1FastDefaults])
steps['ZEEFS']=merge([{'cfg':'ZEE_8TeV_cfi'},K100byK2,step1FastDefaults])
steps['ZTTFS']=merge([{'cfg':'ZTT_Tauola_OneLepton_OtherHadrons_8TeV_cfi'},K100byK2,step1FastDefaults])
steps['QCDFlatPt153000FS']=merge([{'cfg':'QCDForPF_8TeV_cfi'},step1FastDefaults])
steps['QCD_Pt_80_120FS']=merge([{'cfg':'QCD_Pt_80_120_8TeV_cfi'},K100by500,stCond,step1FastDefaults])
steps['QCD_Pt_3000_3500FS']=merge([{'cfg':'QCD_Pt_3000_3500_8TeV_cfi'},K100by500,stCond,step1FastDefaults])
steps['H130GGgluonfusionFS']=merge([{'cfg':'H130GGgluonfusion_8TeV_cfi'},step1FastDefaults])
steps['SingleGammaFlatPt10To10FS']=merge([{'cfg':'SingleGammaFlatPt10To100_cfi'},K100by500,step1FastDefaults])

steps['TTbarSFS']=merge([{'cfg':'TTbar_Tauola_8TeV_cfi'},
                        {'-s':'GEN,SIM',
                         '--eventcontent':'FEVTDEBUG',
                         '--datatier':'GEN-SIM',
                         '--fast':''},
                        step1Defaults])
steps['TTbarSFSA']=merge([{'cfg':'TTbar_Tauola_8TeV_cfi',
                           's':'GEN,SIM,RECO,HLT,VALIDATION',
                           '--fast':''},
                          step1FastDefaults])

def identityFS(wf):
    return merge([{'--restoreRND':'HLT','--process':'HLT2','--hltProcess':'HLT2'},wf])

steps['SingleMuPt10FS_ID']=identityFS(steps['SingleMuPt10FS'])
steps['TTbarFS_ID']=identityFS(steps['TTbarFS'])

#### generator test section ####
step1GenDefaults=merge([{'-s':'GEN,VALIDATION:genvalid',
                         '--relval':'1000000,20000',
                         '--eventcontent':'RAWSIM',
                         '--datatier':'GEN'},
                        step1Defaults])
def genvalid(fragment,d,suffix='all',fi=''):
    import copy
    c=copy.copy(d)
    if suffix:
        c['-s']=c['-s'].replace('genvalid','genvalid_'+suffix)
    if fi:
        c['--filein']='lhe:%d'%(fi,)
    c['cfg']=fragment
    return c
    
steps['QCD_Pt-30_8TeV_herwigpp']=genvalid('QCD_Pt_30_8TeV_herwigpp_cff',step1GenDefaults)
steps['DYToLL_M-50_TuneZ2star_8TeV_pythia6-tauola']=genvalid('DYToLL_M_50_TuneZ2star_8TeV_pythia6_tauola_cff',step1GenDefaults)
steps['QCD_Pt-30_TuneZ2star_8TeV_pythia6']=genvalid('QCD_Pt_30_TuneZ2star_8TeV_pythia6_cff',step1GenDefaults)
steps['QCD_Pt-30_8TeV_pythia8']=genvalid('QCD_Pt_30_8TeV_pythia8_cff',step1GenDefaults)
steps['GluGluTo2Jets_M-100_8TeV_exhume']=genvalid('GluGluTo2Jets_M_100_8TeV_exhume_cff',step1GenDefaults)
steps['TT_TuneZ2star_8TeV_pythia6-evtgen']=genvalid('TT_TuneZ2star_8TeV_pythia6_evtgen_cff',step1GenDefaults)
steps['MinBias_TuneZ2star_8TeV_pythia6']=genvalid('MinBias_TuneZ2star_8TeV_pythia6_cff',step1GenDefaults)
steps['WToLNu_TuneZ2star_8TeV_pythia6-tauola']=genvalid('WToLNu_TuneZ2star_8TeV_pythia6_tauola_cff',step1GenDefaults)
steps['QCD_Pt-30_8TeV_herwig6']=genvalid('QCD_Pt_30_8TeV_herwig6_cff',step1GenDefaults)
steps['MinBias_8TeV_pythia8']=genvalid('MinBias_8TeV_pythia8_cff',step1GenDefaults)


steps['QCD_Ht-100To250_TuneZ2star_8TeV_madgraph-tauola']=genvalid('Hadronizer_MgmMatchTuneZ2star_8TeV_madgraph_tauola_cff',step1GenDefaults,fi=5475)
steps['QCD_Ht-250To500_TuneZ2star_8TeV_madgraph-tauola']=genvalid('Hadronizer_MgmMatchTuneZ2star_8TeV_madgraph_tauola_cff',step1GenDefaults,fi=5476)
steps['QCD_Ht-500To1000_TuneZ2star_8TeV_madgraph-tauola']=genvalid('Hadronizer_MgmMatchTuneZ2star_8TeV_madgraph_tauola_cff',step1GenDefaults,fi=5481)
steps['TTJets_TuneZ2star_8TeV_madgraph-tauola']=genvalid('Hadronizer_MgmMatchTuneZ2star_8TeV_madgraph_tauola_cff',step1GenDefaults,fi=5502)
steps['WJetsLNu_TuneZ2star_8TeV_madgraph-tauola']=genvalid('Hadronizer_MgmMatchTuneZ2star_8TeV_madgraph_tauola_cff',step1GenDefaults,fi=5607)
steps['ZJetsLNu_TuneZ2star_8TeV_madgraph-tauola']=genvalid('Hadronizer_MgmMatchTuneZ2star_8TeV_madgraph_tauola_cff',step1GenDefaults,fi=5591)
steps['ZJetsLNu_Tune4C_8TeV_madgraph-pythia8']=genvalid('Hadronizer_MgmMatchTune4C_8TeV_madgraph_pythia8_cff',step1GenDefaults,fi=5591)

PU={'-n':10,'--pileup':'default','--pileup_input':'dbs:/RelValMinBias/%s/GEN-SIM'%(baseDataSetRelease[0],)}
PUFS={'--pileup':'default'}
PUFS2={'--pileup':'mix_2012_Startup_inTimeOnly'}
steps['TTbarFSPU']=merge([PUFS,steps['TTbarFS']])
steps['TTbarFSPU2']=merge([PUFS2,steps['TTbarFS']])
##########################



# step2 
step2Defaults = { 
                  '-s'            : 'DIGI,L1,DIGI2RAW,HLT:@relval,RAW2DIGI,L1Reco',
                  '--datatier'    : 'GEN-SIM-DIGI-RAW-HLTDEBUG',
                  '--eventcontent': 'FEVTDEBUGHLT',
                  '--conditions'  : 'auto:startup',
                  }


steps['DIGIPROD1']=merge([{'--eventcontent':'RAWSIM','--datatier':'GEN-SIM-RAW'},step2Defaults])
steps['DIGI']=merge([step2Defaults])
#steps['DIGI2']=merge([stCond,step2Defaults])
steps['DIGICOS']=merge([{'--scenario':'cosmics','--eventcontent':'FEVTDEBUG','--datatier':'GEN-SIM-DIGI-RAW'},stCond,step2Defaults])

steps['DIGIPU1']=merge([PU,step2Defaults])
steps['REDIGIPU']=merge([{'-s':'reGEN,reDIGI,L1,DIGI2RAW,HLT:@relval,RAW2DIGI,L1Reco'},steps['DIGIPU1']])

steps['DIGI_ID']=merge([{'--restoreRND':'HLT','--process':'HLT2'},steps['DIGI']])

steps['RESIM']=merge([{'-s':'reGEN,reSIM','-n':10},steps['DIGI']])
steps['RESIMDIGI']=merge([{'-s':'reGEN,reSIM,DIGI,L1,DIGI2RAW,HLT:@relval,RAW2DIGI,L1Reco','-n':10,'--restoreRNDSeeds':'','--process':'HLT'},steps['DIGI']])

    
steps['DIGIHI']=merge([{'--conditions':'auto:starthi_HIon', '-s':'DIGI,L1,DIGI2RAW,HLT:HIon,RAW2DIGI,L1Reco', '--inputCommands':'"keep *","drop *_simEcalPreshowerDigis_*_*"', '-n':10}, hiDefaults, step2Defaults])

#add this line when testing from an input file that is not strictly GEN-SIM
#addForAll(step2,{'--process':'DIGI'})

dataReco={'--conditions':'auto:com10',
          '-s':'RAW2DIGI,L1Reco,RECO,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias,DQM',
          '--datatier':'RECO,DQMROOT',
          '--eventcontent':'RECO,DQMROOT',
          '--data':'',
          '--process':'reRECO',
          '--scenario':'pp',
          }

hltKey='relval'
from Configuration.HLT.autoHLT import autoHLT
menu = autoHLT[hltKey]
steps['HLTD']=merge([{'--process':'reHLT',
                      '-s':'L1REPACK,HLT:@%s'%hltKey,
                      '--conditions':'auto:hltonline_%s'%menu,
                      '--data':'',
                      '--output':'\'[{"e":"RAW","t":"RAW","o":["drop FEDRawDataCollection_rawDataCollector__LHC"]}]\'',
                      },])
wmsplit['HLTD']=5

steps['RECOD']=merge([{'--scenario':'pp',},dataReco])
steps['RECOSKIMALCA']=merge([{'--inputCommands':'"keep *","drop *_*_*_RECO"'
                              },steps['RECOD']])
steps['RECOSKIM']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,DQM',
                          },steps['RECOSKIMALCA']])

steps['REPACKHID']=merge([{'--scenario':'HeavyIons',
                         '-s':'RAW2DIGI,REPACK',
                         '--datatier':'RAW',
                         '--eventcontent':'REPACKRAW'},
                        steps['RECOD']])
steps['RECOHID10']=merge([{'--scenario':'HeavyIons',
                         '-s':'RAW2DIGI,L1Reco,RECO,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBiasHI+HcalCalMinBias,DQM',
                         '--datatier':'RECO,DQMROOT',
                         '--eventcontent':'RECO,DQMROOT'},
                        steps['RECOD']])
steps['RECOHID11']=merge([{'--repacked':''},
                        steps['RECOHID10']])
steps['RECOHID10']['-s']+=',REPACK'
steps['RECOHID10']['--datatier']+=',RAW'
steps['RECOHID10']['--eventcontent']+=',REPACKRAW'

steps['TIER0']=merge([{'--customise':'Configuration/DataProcessing/RecoTLR.customisePrompt',
                       '-s':'RAW2DIGI,L1Reco,RECO,ALCAPRODUCER:@allForPrompt,DQM,ENDJOB',
                       '--datatier':'RECO,AOD,ALCARECO,DQMROOT',
                       '--eventcontent':'RECO,AOD,ALCARECO,DQMROOT',
                       '--process':'RECO'
                       },dataReco])
steps['TIER0EXP']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCAPRODUCER:@allForExpress,DQM,ENDJOB',
                          '--datatier':'ALCARECO,DQM',
                          '--eventcontent':'ALCARECO,DQM',
                          '--customise':'Configuration/DataProcessing/RecoTLR.customiseExpress',
                          },steps['TIER0']])

steps['RECOCOSD']=merge([{'--scenario':'cosmics',
                          '-s':'RAW2DIGI,L1Reco,RECO,DQM,ALCA:MuAlCalIsolatedMu+DtCalib',
                          '--customise':'Configuration/DataProcessing/RecoTLR.customiseCosmicData'
                          },dataReco])

step2HImixDefaults=merge([{'-n':'10',
                           '--himix':'',
                           '--filein':'file.root',
                           '--process':'HISIGNAL'
                           },hiDefaults,step1Defaults])
steps['Pyquen_GammaJet_pt20_2760GeV']=merge([{'cfg':'Pyquen_GammaJet_pt20_2760GeV_cfi'},step2HImixDefaults])
steps['Pyquen_DiJet_pt80to120_2760GeV']=merge([{'cfg':'Pyquen_DiJet_pt80to120_2760GeV_cfi'},step2HImixDefaults])
steps['Pyquen_ZeemumuJets_pt10_2760GeV']=merge([{'cfg':'Pyquen_ZeemumuJets_pt10_2760GeV_cfi'},step2HImixDefaults])

# step3 
step3Defaults = {
                  '-s'            : 'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                  '--conditions'  : 'auto:startup',
                  '--no_exec'     : '',
                  '--datatier'    : 'GEN-SIM-RECO,DQM',
                  '--eventcontent': 'RECOSIM,DQM'
                  }

steps['DIGIPU']=merge([{'--process':'REDIGI'},steps['DIGIPU1']])

steps['RECODreHLT']=merge([{'--hltProcess':'reHLT','--conditions':'auto:com10_%s'%menu},steps['RECOD']])
wmsplit['RECODreHLT']=2

steps['RECO']=merge([step3Defaults])
steps['RECODBG']=merge([{'--eventcontent':'RECODEBUG,DQM'},steps['RECO']])
steps['RECOPROD1']=merge([{ '-s' : 'RAW2DIGI,L1Reco,RECO', '--datatier' : 'GEN-SIM-RECO,AODSIM', '--eventcontent' : 'RECOSIM,AODSIM'},step3Defaults])
steps['RECOCOS']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCA:MuAlCalIsolatedMu,DQM','--scenario':'cosmics'},stCond,step3Defaults])
steps['RECOMIN']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCA:SiStripCalZeroBias+SiStripCalMinBias+EcalCalPhiSym+EcalCalPi0Calib+EcalCalEtaCalib,VALIDATION,DQM'},stCond,step3Defaults])

steps['RECODDQM']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,DQM:@common+@muon+@hcal+@jetmet+@ecal'},steps['RECOD']])

steps['RECOPU1']=merge([PU,steps['RECO']])
steps['RECOPUDBG']=merge([{'--eventcontent':'RECODEBUG,DQM'},steps['RECOPU1']])
steps['RERECOPU1']=merge([{'--hltProcess':'REDIGI'},steps['RECOPU1']])

steps['RECO_ID']=merge([{'--hltProcess':'HLT2'},steps['RECO']])

steps['RECOHI']=merge([hiDefaults,step3Defaults])
steps['DIGIHISt3']=steps['DIGIHI']

steps['RECOHID11St3']=merge([{
                              '--process':'ZStoRECO'},
                             steps['RECOHID11']])
steps['RECOHIR10D11']=merge([{'--filein':'file:step2_inREPACKRAW.root',
                              '--filtername':'reRECO'},
                             steps['RECOHID11St3']])
steps['RECOFS']=merge([{'--fast':'',
                        '-s':'RECO,HLT:@relval,VALIDATION'},
                       steps['RECO']])

#add this line when testing from an input file that is not strictly GEN-SIM
#addForAll(step3,{'--hltProcess':'DIGI'})

steps['ALCACOSD']={'--conditions':'auto:com10',
                   '--datatier':'ALCARECO',
                   '--eventcontent':'ALCARECO',
                   '--scenario':'cosmics',
                   '-s':'ALCA:TkAlCosmics0T+MuAlGlobalCosmics+HcalCalHOCosmics+DQM'
                   }
steps['ALCAPROMPT']={'-s':'ALCA:PromptCalibProd',
                     '--filein':'file:TkAlMinBias.root',
                     '--conditions':'auto:com10',
                     '--datatier':'ALCARECO',
                     '--eventcontent':'ALCARECO'}
steps['ALCAEXP']={'-s':'ALCA:PromptCalibProd',
                  '--conditions':'auto:com10',
                  '--datatier':'ALCARECO',
                  '--eventcontent':'ALCARECO'}

# step4
step4Defaults = { 
                  '-s'            : 'ALCA:TkAlMuonIsolated+TkAlMinBias+EcalCalElectron+HcalCalIsoTrk+MuAlOverlaps',
                  '-n'            : 1000,
                  '--conditions'  : 'auto:startup',
                  '--datatier'    : 'ALCARECO',
                  '--eventcontent': 'ALCARECO',
                  }

steps['RERECOPU']=steps['RERECOPU1']

steps['ALCATT']=merge([{'--filein':'file:step3.root'},step4Defaults])
steps['ALCAMIN']=merge([{'-s':'ALCA:TkAlMinBias','--filein':'file:step3.root'},stCond,step4Defaults])
steps['ALCACOS']=merge([{'-s':'ALCA:TkAlCosmics0T+MuAlGlobalCosmics+HcalCalHOCosmics'},stCond,step4Defaults])
steps['ALCABH']=merge([{'-s':'ALCA:TkAlBeamHalo+MuAlBeamHaloOverlaps+MuAlBeamHalo'},stCond,step4Defaults])
steps['ALCAELE']=merge([{'-s':'ALCA:EcalCalElectron','--filein':'file:step3.root'},stCond,step4Defaults])

steps['ALCAHARVD']={'-s':'ALCAHARVEST:BeamSpotByRun+BeamSpotByLumi+SiStripQuality',
                    '--conditions':'auto:com10',
                    '--scenario':'pp',
                    '--data':'',
                    '--filein':'file:PromptCalibProd.root'}

steps['RECOHISt4']=steps['RECOHI']

steps['ALCANZS']=merge([{'-s':'ALCA:HcalCalMinBias','--mc':''},step4Defaults])
steps['HARVGEN']={'-s':'HARVESTING:genHarvesting',
                  '--harvesting':'AtJobEnd',
                  '--conditions':'auto:startup',
                  '--mc':'',
                  '--filein':'file:step1.root'
                  }

#data
steps['HARVESTD']={'-s':'HARVESTING:dqmHarvesting',
                   '--conditions':'auto:com10',
                   '--filetype':'DQM',
                   '--data':'',
                   '--scenario':'pp'}

steps['HARVESTDreHLT'] = merge([ {'--conditions':'auto:com10_%s'%menu}, steps['HARVESTD'] ])

steps['HARVESTDDQM']=merge([{'-s':'HARVESTING:@common+@muon+@hcal+@jetmet+@ecal'},steps['HARVESTD']])

steps['HARVESTDfst2']=merge([{'--filein':'file:step2_inDQM.root'},steps['HARVESTD']])

steps['HARVESTDC']={'-s':'HARVESTING:dqmHarvesting',
                   '--conditions':'auto:com10',
                   '--filetype':'DQM',
                   '--data':'',
                    '--filein':'file:step2_inDQM.root',
                   '--scenario':'cosmics'}
steps['HARVESTDHI']={'-s':'HARVESTING:dqmHarvesting',
                   '--conditions':'auto:com10',
                   '--filetype':'DQM',
                   '--data':'',
                   '--scenario':'HeavyIons'}

#MC
steps['HARVEST']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                   '--conditions':'auto:startup',
                   '--mc':'',
                   '--scenario':'pp'}
steps['HARVESTCOS']={'-s':'HARVESTING:dqmHarvesting',
                     '--conditions':'auto:startup',
                     '--mc':'',
                     '--filein':'file:step3_inDQM.root',
                   '--scenario':'cosmics'}
steps['HARVESTFS']={'-s':'HARVESTING:validationHarvestingFS',
                   '--conditions':'auto:startup',
                   '--mc':'',
                   '--scenario':'pp'}
steps['HARVESTHI']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                   '--conditions':'auto:starthi_HIon',
                   '--mc':'',
                   '--scenario':'HeavyIons'}

steps['ALCASPLIT']={'-s':'ALCAOUTPUT:@allForPrompt',
                    '--conditions':'auto:com10',
                    '--scenario':'pp',
                    '--data':'',
                    '--triggerResultsProcess':'RECO',
                    '--filein':'file:step2_inALCARECO.root'}

steps['SKIMD']={'-s':'SKIM:all',
                '--conditions':'auto:com10',
                '--data':'',
                '--scenario':'pp',
                '--filein':'file:step2.root',
                '--secondfilein':'filelist:step1_dbsquery.log'}

steps['SKIMDreHLT'] = merge([ {'--conditions':'auto:com10_%s'%menu,'--filein':'file:step3.root'}, steps['SKIMD'] ])

steps['SKIMCOSD']={'-s':'SKIM:all',
                   '--conditions':'auto:com10',
                   '--data':'',
                   '--scenario':'cosmics',
                   '--filein':'file:step2.root',
                   '--secondfilein':'filelist:step1_dbsquery.log'}
                 

#### for special wfs ###
#steps['TTbar_REDIGI_RERECO']=merge([{'cfg':'TTbar_Tauola_8TeV_cfi',
#                                     '-s':'GEN,SIM,DIGI,L1,DIGI2RAW,HLT:@relval,RAW2DIGI,L1Reco,RECO,ALCA:MuAlCalIsolatedMu+DtCalib,VALIDATION,DQM',
#                                     '--datatier':'GEN-SIM-DIGI-RAW-HLTDEBUG-RECO,DQM',
#                                     '--eventcontent':'FEVTDEBUGHLT,DQM'},
#                                    K9by50,stCond,step1Defaults])
#steps['DIGI2RECO']=merge([{'-s':'DIGI,L1,DIGI2RAW,HLT:@relval,RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
#                           '--filtername':'DIGItoRECO',
#                           '--process':'RECO',
#                           '--eventcontent':'RECOSIM,DQM',
#                           '--datatier':'GEN-SIM-RECO,DQM',
#                           },
#                            stCond,step3Defaults])
steps['RECOFROMRECO']=merge([{'-s':'RECO',
                              '--filtername':'RECOfromRECO',
                              '--process':'reRECO',
                              '--datatier':'AODSIM',
                              '--eventcontent':'AODSIM',
                              },
                             stCond,step3Defaults])


steps['RECOFROMRECOSt2']=steps['RECOFROMRECO']

steps['RECODFROMRAWRECO']=merge([{'-s':'RAW2DIGI:RawToDigi_noTk,L1Reco,RECO:reconstruction_noTracking',
                                  '--filtername':'RECOfromRAWRECO',
                                  '--process':'rereRECO',
                                  '--datatier':'AOD',
                                  '--eventcontent':'AOD',
                                  '--secondfilein':'filelist:step1_dbsquery.log',
                                  },
                                 steps['RECOD']])


steps['COPYPASTE']={'-s':'NONE',
                    '--conditions':'auto:startup',
                    '--output':'\'[{"t":"RAW","e":"ALL"}]\'',
                    '--customise_commands':'"process.ALLRAWoutput.fastCloning=cms.untracked.bool(False)"'}
