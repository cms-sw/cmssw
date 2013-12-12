

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
    def __init__(self,dataSet,label='',run=[],files=1000,events=InputInfoNDefault,split=10,location='CAF',ib_blacklist=None,ib_block=None) :
        self.run = run
        self.files = files
        self.events = events
        self.location = location
        self.label = label
        self.dataSet = dataSet
        self.split = split
        self.ib_blacklist = ib_blacklist
        self.ib_block = ib_block
        
    def dbs(self):
        query_by = "block" if self.ib_block else "dataset"
        query_source = "{0}#{1}".format(self.dataSet, self.ib_block) if self.ib_block else self.dataSet
        if len(self.run) is not 0:
            command = ";".join(["das_client.py --limit=0 --query 'file {0}={1} run={2}'".format(query_by, query_source, query_run) for query_run in self.run])
            command = "({0})".format(command)
        else:
            command = "das_client.py --limit=0 --query 'file {0}={1} site=T2_CH_CERN'".format(query_by, query_source)
       
        # Run filter on DAS output 
        if self.ib_blacklist:
            command += " | grep -E -v "
            command += " ".join(["-e '{0}'".format(pattern) for pattern in self.ib_blacklist])
        command += " | sort -u"
        return command

    def lumiRanges(self):
        if len(self.run) != 0:
            return "echo '{\n"+",".join(('"%d":[[1,268435455]]\n'%(x,) for x in self.run))+"}'"
        return None

    def __str__(self):
        if self.ib_block:
            return "input from: {0} with run {1}#{2}".format(self.dataSet, self.ib_block, self.run)
        return "input from: {0} with run {1}".format(self.dataSet, self.run)
    
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
#wmsplit = {}

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
steps['ValSkim2011A']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2011A-ValSkim-08Nov2011-v1/RAW-RECO',ib_block='239c497e-0fae-11e1-a8b1-00221959e72f',label='run2011A',location='STD',run=Run2011ASk)}
steps['WMuSkim2011A']={'INPUT':InputInfo(dataSet='/SingleMu/Run2011A-WMu-08Nov2011-v1/RAW-RECO',ib_block='388c2990-0de6-11e1-bb7e-00221959e72f',label='wMu2011A',location='STD',run=Run2011ASk)}
steps['WElSkim2011A']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2011A-WElectron-08Nov2011-v1/RAW-RECO',ib_block='9c48c4ea-0db2-11e1-b62c-00221959e69e',label='wEl2011A',location='STD',run=Run2011ASk)}
steps['ZMuSkim2011A']={'INPUT':InputInfo(dataSet='/DoubleMu/Run2011A-ZMu-08Nov2011-v1/RAW-RECO',label='zMu2011A',location='STD',run=Run2011ASk)}
steps['ZElSkim2011A']={'INPUT':InputInfo(dataSet='/DoubleElectron/Run2011A-ZElectron-08Nov2011-v1/RAW-RECO',label='zEl2011A',location='STD',run=Run2011ASk)}
steps['HighMet2011A']={'INPUT':InputInfo(dataSet='/Jet/Run2011A-HighMET-08Nov2011-v1/RAW-RECO',ib_block='3c764584-0b59-11e1-b62c-00221959e69e',label='hMet2011A',location='STD',run=Run2011ASk)}

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

steps['ValSkim2011B']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2011B-ValSkim-19Nov2011-v1/RAW-RECO',label='run2011B',location='STD',run=Run2011BSk)}
steps['WMuSkim2011B']={'INPUT':InputInfo(dataSet='/SingleMu/Run2011B-WMu-19Nov2011-v1/RAW-RECO',ib_block='19110c74-1b66-11e1-a98b-003048f02c8a',label='wMu2011B',location='STD',run=Run2011BSk)}
steps['WElSkim2011B']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2011B-WElectron-19Nov2011-v1/RAW-RECO',ib_block='d75771a4-1b3f-11e1-aef4-003048f02c8a',label='wEl2011B',location='STD',run=Run2011BSk)}
steps['ZMuSkim2011B']={'INPUT':InputInfo(dataSet='/DoubleMu/Run2011B-ZMu-19Nov2011-v1/RAW-RECO',label='zMu2011B',location='STD',run=Run2011BSk)}
steps['ZElSkim2011B']={'INPUT':InputInfo(dataSet='/DoubleElectron/Run2011B-ZElectron-19Nov2011-v1/RAW-RECO',label='zEl2011B',run=Run2011BSk)}
steps['HighMet2011B']={'INPUT':InputInfo(dataSet='/Jet/Run2011B-HighMET-19Nov2011-v1/RAW-RECO',label='hMet2011B',run=Run2011BSk)}

steps['RunHI2010']={'INPUT':InputInfo(dataSet='/HIAllPhysics/HIRun2010-v1/RAW',label='hi2010',run=[152698],events=10000,location='STD')}
steps['RunHI2011']={'INPUT':InputInfo(dataSet='/HIAllPhysics/HIRun2011A-v1/RAW',label='hi2011',run=[174773],events=10000,location='STD')}


Run2012A=[191226]
Run2012ASk=Run2012A+[]
steps['RunMinBias2012A']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2012A-v1/RAW',label='mb2012A',run=Run2012A, events=100000,location='STD')}
steps['RunTau2012A']={'INPUT':InputInfo(dataSet='/Tau/Run2012A-v1/RAW',label='tau2012A', run=Run2012A, events=100000,location='STD')}
steps['RunMET2012A']={'INPUT':InputInfo(dataSet='/MET/Run2012A-v1/RAW',label='met2012A', run=Run2012A, events=100000,location='STD')}
steps['RunMu2012A']={'INPUT':InputInfo(dataSet='/SingleMu/Run2012A-v1/RAW',label='mu2012A', run=Run2012A, events=100000,location='STD')}
steps['RunElectron2012A']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2012A-v1/RAW',label='electron2012A', run=Run2012A, events=100000,location='STD')}
steps['RunJet2012A']={'INPUT':InputInfo(dataSet='/Jet/Run2012A-v1/RAW',label='jet2012A', run=Run2012A, events=100000,location='STD')}

steps['WElSkim2012A']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2012A-WElectron-13Jul2012-v1/USER',label='wEl2012A',location='STD',run=Run2012ASk)}
steps['ZMuSkim2012A']={'INPUT':InputInfo(dataSet='/SingleMu/Run2012A-ZMu-13Jul2012-v1/RAW-RECO',label='zMu2012A',location='STD',run=Run2012ASk)}
steps['ZElSkim2012A']={'INPUT':InputInfo(dataSet='/DoubleElectron/Run2012A-ZElectron-13Jul2012-v1/RAW-RECO',label='zEl2012A',run=Run2012ASk)}
steps['HighMet2012A']={'INPUT':InputInfo(dataSet='/HT/Run2012A-HighMET-13Jul2012-v1/RAW-RECO',label='hMet2012A',run=Run2012ASk)}


Run2012B=[194533]
Run2012Bsk=Run2012B+[194912,195016]
steps['RunMinBias2012B']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2012B-v1/RAW',label='mb2012B',run=Run2012B, events=100000,location='STD')}
steps['RunMu2012B']={'INPUT':InputInfo(dataSet='/SingleMu/Run2012B-v1/RAW',label='mu2012B',location='STD',run=Run2012B)}
steps['RunPhoton2012B']={'INPUT':InputInfo(dataSet='/SinglePhoton/Run2012B-v1/RAW',ib_block='28d7fcc8-a2a0-11e1-86c7-003048caaace',label='photon2012B',location='STD',run=Run2012B)}
steps['RunEl2012B']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2012B-v1/RAW',label='electron2012B',location='STD',run=Run2012B)}
steps['RunJet2012B']={'INPUT':InputInfo(dataSet='/JetHT/Run2012B-v1/RAW',label='jet2012B',location='STD',run=Run2012B)}
steps['ZMuSkim2012B']={'INPUT':InputInfo(dataSet='/SingleMu/Run2012B-ZMu-13Jul2012-v1/RAW-RECO',label='zMu2012B',location='CAF',run=Run2012Bsk)}
steps['WElSkim2012B']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2012B-WElectron-13Jul2012-v1/USER',label='wEl2012B',location='STD',run=Run2012Bsk)}
steps['ZElSkim2012B']={'INPUT':InputInfo(dataSet='/DoubleElectron/Run2012B-ZElectron-13Jul2012-v1/RAW-RECO',ib_block='0c5092cc-d554-11e1-bb62-00221959e69e',label='zEl2012B',location='STD',run=Run2012Bsk)}

Run2012C=[199812]
Run2012Csk=Run2012C+[203002]
steps['RunMinBias2012C']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2012C-v1/RAW',label='mb2012C',run=Run2012C, events=100000,location='STD')}
steps['RunMu2012C']={'INPUT':InputInfo(dataSet='/SingleMu/Run2012C-v1/RAW',label='mu2012C',location='STD',run=Run2012C)}
steps['RunPhoton2012C']={'INPUT':InputInfo(dataSet='/SinglePhoton/Run2012C-v1/RAW',label='photon2012C',location='STD',run=Run2012C)}
steps['RunEl2012C']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2012C-v1/RAW',label='electron2012C',location='STD',run=Run2012C)}
steps['RunJet2012C']={'INPUT':InputInfo(dataSet='/JetHT/Run2012C-v1/RAW',label='jet2012C',location='STD',run=Run2012C)}
steps['ZMuSkim2012C']={'INPUT':InputInfo(dataSet='/SingleMu/Run2012C-ZMu-PromptSkim-v3/RAW-RECO',label='zMu2012C',location='CAF',run=Run2012Csk)}
steps['WElSkim2012C']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2012C-WElectron-PromptSkim-v3/USER',label='wEl2012C',location='STD',run=Run2012Csk)}
steps['ZElSkim2012C']={'INPUT':InputInfo(dataSet='/DoubleElectron/Run2012C-ZElectron-PromptSkim-v3/RAW-RECO',label='zEl2012C',location='STD',run=Run2012Csk)}

Run2012D=[208307]
Run2012Dsk=Run2012D+[207454]
steps['RunMinBias2012D']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2012D-v1/RAW',label='mb2012D',run=Run2012D, events=100000,location='STD')}
steps['RunMu2012D']={'INPUT':InputInfo(dataSet='/SingleMu/Run2012D-v1/RAW',label='mu2012D',location='STD',run=Run2012D)}
steps['RunPhoton2012D']={'INPUT':InputInfo(dataSet='/SinglePhoton/Run2012D-v1/RAW',label='photon2012D',location='STD',run=Run2012D)}
steps['RunEl2012D']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2012D-v1/RAW',label='electron2012D',location='STD',run=Run2012D)}
steps['RunJet2012D']={'INPUT':InputInfo(dataSet='/JetHT/Run2012D-v1/RAW',label='jet2012D',location='STD',run=Run2012D)}
steps['ZMuSkim2012D']={'INPUT':InputInfo(dataSet='/SingleMu/Run2012D-ZMu-PromptSkim-v1/RAW-RECO',label='zMu2012D',location='STD',run=Run2012Dsk)}
steps['WElSkim2012D']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2012D-WElectron-PromptSkim-v1/USER',label='wEl2012D',location='STD',run=Run2012Dsk)}
steps['ZElSkim2012D']={'INPUT':InputInfo(dataSet='/DoubleElectron/Run2012D-ZElectron-PromptSkim-v1/RAW-RECO',label='zEl2012D',location='STD',run=Run2012Dsk)}

#### Standard release validation samples ####

stCond={'--conditions':'auto:startup'}
def Kby(N,s):
    return {'--relval':'%s000,%s'%(N,s)}
def Mby(N,s):
    return {'--relval':'%s000000,%s'%(N,s)}

def gen(fragment,howMuch):
    global step1Defaults
    return merge([{'cfg':fragment},howMuch,step1Defaults])

steps['MinBias']=gen('MinBias_8TeV_cfi',Kby(9,300))
steps['QCD_Pt_3000_3500']=gen('QCD_Pt_3000_3500_8TeV_cfi',Kby(9,25))
steps['QCD_Pt_600_800']=gen('QCD_Pt_600_800_8TeV_cfi',Kby(9,50))
steps['QCD_Pt_80_120']=gen('QCD_Pt_80_120_8TeV_cfi',Kby(9,100))
steps['QCD_Pt_30_80_BCtoE_8TeV']=gen('QCD_Pt_30_80_BCtoE_8TeV',Kby(9000,100))
steps['QCD_Pt_80_170_BCtoE_8TeV']=gen('QCD_Pt_80_170_BCtoE_8TeV',Kby(9000,100))
steps['SingleElectronPt10']=gen('SingleElectronPt10_cfi',Kby(9,3000))
steps['SingleElectronPt35']=gen('SingleElectronPt35_cfi',Kby(9,500))
steps['SingleElectronPt1000']=gen('SingleElectronPt1000_cfi',Kby(9,50))
steps['SingleElectronFlatPt1To100']=gen('SingleElectronFlatPt1To100_cfi',Mby(2,100))
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
steps['ZpMM_2250_8TeV_Tauola']=gen('ZpMM_2250_8TeV_Tauola_cfi',Kby(9,100))
steps['ZpEE_2250_8TeV_Tauola']=gen('ZpEE_2250_8TeV_Tauola_cfi',Kby(9,100))
steps['ZpTT_1500_8TeV_Tauola']=gen('ZpTT_1500_8TeV_Tauola_cfi',Kby(9,100))



def identitySim(wf):
    return merge([{'--restoreRND':'SIM','--process':'SIM2'},wf])

steps['SingleMuPt10_ID']=identitySim(steps['SingleMuPt10'])
steps['TTbar_ID']=identitySim(steps['TTbar'])

baseDataSetRelease=[
    'CMSSW_6_2_0_pre2-START61_V11_g496p1-v1',#'CMSSW_6_1_0_pre6-START61_V5-v1',#'CMSSW_6_0_0-START60_V4-v1',
    'CMSSW_6_2_0_pre2-STARTHI61_V13_g496p1-v1',#'CMSSW_6_1_0_pre6-STARTHI61_V6-v1',#'CMSSW_6_0_0-STARTHI60_V4-v1',
    'CMSSW_6_2_0_pre2-START61_V11_g496p1-v2',#'CMSSW_6_1_0_pre6-START61_V5-v2',#'CMSSW_6_0_0-PU_START60_V4-v1',
    'CMSSW_6_1_0_pre6-START61_V5_FastSim-v1',#'CMSSW_6_0_0-START60_V4_FastSim-v1'
    'CMSSW_6_2_0_pre2-START61_V11_g496p1-v3',
    'CMSSW_6_2_0_pre2-START61_V11_g496p1_02May2013-v1' ## this is a fuck up in the dataset naming
    ]

steps['MinBiasINPUT']={'INPUT':InputInfo(dataSet='/RelValMinBias/%s/GEN-SIM'%(baseDataSetRelease[2],),location='STD')}
steps['QCD_Pt_3000_3500INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_3000_3500/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['QCD_Pt_600_800INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_600_800/%s/GEN-SIM'%(baseDataSetRelease[2],),location='STD')}
steps['QCD_Pt_80_120INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_80_120/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['SingleElectronPt10INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleElectronPt10/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['SingleElectronPt1000INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleElectronPt1000/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['SingleElectronPt35INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleElectronPt35/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['SingleGammaPt10INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleGammaPt10/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['SingleGammaPt35INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleGammaPt35/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['SingleMuPt1INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt1/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['SingleMuPt10INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt10/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['SingleMuPt10IdINPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt10/%s/GEN-SIM-DIGI-RAW-HLTDEBUG'%(baseDataSetRelease[0],),location='STD',split=1)}
steps['SingleMuPt10FSIdINPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt10/%s/GEN-SIM-DIGI-RECO'%(baseDataSetRelease[3],),location='STD',split=1)}
steps['SingleMuPt100INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt100/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['SingleMuPt1000INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt1000/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['TTbarINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['TTbarIdINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/%s/GEN-SIM-DIGI-RAW-HLTDEBUG'%(baseDataSetRelease[0],),location='STD',split=1)}
steps['TTbarFSIdINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/%s/GEN-SIM-DIGI-RECO'%(baseDataSetRelease[3],),location='STD',split=1)}
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
ecalHcal={
    '-s':'GEN,SIM,DIGI,DIGI2RAW,RAW2DIGI,L1Reco,RECO,EI',
    '--datatier':'GEN-SIM-DIGI-RAW-RECO',
    #'--geometry':'ECALHCAL',
    '--eventcontent':'FEVTDEBUG',
    '--customise':'Validation/Configuration/ECALHCAL.customise',
    '--beamspot':'NoSmear'}

steps['SingleElectronE120EHCAL']=merge([{'cfg':'SingleElectronE120EHCAL_cfi'},ecalHcal,Kby(25,250),step1Defaults])
steps['SinglePiE50HCAL']=merge([{'cfg':'SinglePiE50HCAL_cfi'},ecalHcal,Kby(25,250),step1Defaults])

steps['MinBiasHS']=gen('MinBias_8TeV_cfi',Kby(25,300))
steps['InclusiveppMuX']=gen('InclusiveppMuX_8TeV_cfi',Mby(11,45000))
steps['SingleElectronFlatPt5To100']=gen('SingleElectronFlatPt5To100_cfi',Kby(25,250))
steps['SinglePiPt1']=gen('SinglePiPt1_cfi',Kby(25,250))
steps['SingleMuPt1HS']=gen('SingleMuPt1_cfi',Kby(25,1000))
steps['ZPrime5000Dijet']=gen('ZPrime5000JJ_8TeV_cfi',Kby(25,100))
steps['SinglePi0E10']=gen('SinglePi0E10_cfi',Kby(25,100))
steps['SinglePiPt10']=gen('SinglePiPt10_cfi',Kby(25,250))
steps['SingleGammaFlatPt10To100']=gen('SingleGammaFlatPt10To100_cfi',Kby(25,250))
steps['SingleTauPt50Pythia']=gen('SingleTaupt_50_cfi',Kby(25,100))
steps['SinglePiPt100']=gen('SinglePiPt100_cfi',Kby(25,250))


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
steps['ZmumuJets_Pt_20_300']=gen('ZmumuJets_Pt_20_300_GEN_8TeV_cfg',Kby(25,100))
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
steps['PhotonJets_Pt_10INPUT']={'INPUT':InputInfo(dataSet='/RelValPhotonJets_Pt_10/%s/GEN-SIM'%(baseDataSetRelease[2],),location='STD')}
steps['QQH1352T_TauolaINPUT']={'INPUT':InputInfo(dataSet='/RelValQQH1352T_Tauola/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['ADDMonoJet_d3MD3INPUT']={'INPUT':InputInfo(dataSet='/RelValADDMonoJet_d3MD3/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['WpMINPUT']={'INPUT':InputInfo(dataSet='/RelValWpM/%s/GEN-SIM'%(baseDataSetRelease[2],),location='STD')}
steps['ZpMMINPUT']={'INPUT':InputInfo(dataSet='/RelValZpMM/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['ZpMM_2250_8TeV_TauolaINPUT']={'INPUT':InputInfo(dataSet='/RelValZpMM_2250_8TeV_Tauola/%s/GEN-SIM'%(baseDataSetRelease[5],),location='STD')}
steps['ZpEE_2250_8TeV_TauolaINPUT']={'INPUT':InputInfo(dataSet='/RelValZpEE_2250_8TeV_Tauola/%s/GEN-SIM'%(baseDataSetRelease[5],),location='STD')}
steps['ZpTT_1500_8TeV_TauolaINPUT']={'INPUT':InputInfo(dataSet='/RelValZpTT_1500_8TeV_Tauola/%s/GEN-SIM'%(baseDataSetRelease[5],),location='STD')}


steps['ZmumuJets_Pt_20_300INPUT']={'INPUT':InputInfo(dataSet='/RelValZmumuJets_Pt_20_300/%s/GEN-SIM'%(baseDataSetRelease[4],),location='STD')}
                                

steps['Cosmics']=merge([{'cfg':'UndergroundCosmicMu_cfi.py','--scenario':'cosmics'},Kby(666,100000),step1Defaults])
steps['BeamHalo']=merge([{'cfg':'BeamHalo_cfi.py','--scenario':'cosmics'},Kby(9,100),step1Defaults])

steps['CosmicsINPUT']={'INPUT':InputInfo(dataSet='/RelValCosmics/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['BeamHaloINPUT']={'INPUT':InputInfo(dataSet='/RelValBeamHalo/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}

steps['QCD_Pt_50_80']=genS('QCD_Pt_50_80_8TeV_cfi',Kby(25,100))
steps['QCD_Pt_15_20']=genS('QCD_Pt_15_20_8TeV_cfi',Kby(25,100))
steps['ZTTHS']=merge([Kby(25,100),steps['ZTT']])
steps['QQH120Inv']=genS('QQH120Inv_8TeV_cfi',Kby(25,100))
steps['TTbar2HS']=merge([Kby(25,100),steps['TTbar']])
steps['JpsiMM_Pt_20_inf']=genS('JpsiMM_Pt_20_inf_8TeV_cfi',Kby(70,280))
steps['QCD_Pt_120_170']=genS('QCD_Pt_120_170_8TeV_cfi',Kby(25,100))
steps['H165WW2L']=genS('H165WW2L_Tauola_8TeV_cfi',Kby(25,100))
steps['UpsMM']=genS('UpsMM_8TeV_cfi',Kby(56250,225))
steps['RSGrav']=genS('RS750_quarks_and_leptons_8TeV_cff',Kby(25,100))
steps['QCD_Pt_80_120_2HS']=merge([Kby(25,100),steps['QCD_Pt_80_120']])
steps['bJpsiX']=genS('bJpsiX_8TeV_cfi',Mby(325,1300000))
steps['QCD_Pt_30_50']=genS('QCD_Pt_30_50_8TeV_cfi',Kby(25,100))
steps['H200ZZ4L']=genS('H200ZZ4L_Tauola_8TeV_cfi',Kby(25,100))
steps['LM9p']=genS('LM9p_8TeV_cff',Kby(25,100))
steps['QCD_Pt_20_30']=genS('QCD_Pt_20_30_8TeV_cfi',Kby(25,100))
steps['QCD_Pt_170_230']=genS('QCD_Pt_170_230_8TeV_cfi',Kby(25,100))



## upgrade dedicated wf

step1Up2017Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'auto:upgrade2017', 
                             '--beamspot' : 'Gauss',
			     '--magField' : '38T_PostLS1',
			     '--datatier' : 'GEN-SIM',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'Extended2017',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017'
                             }
def gen2017(fragment,howMuch):
    global step1Up2017Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2017Defaults])

steps['FourMuPt1_200_UPG2017']=gen2017('FourMuPt_1_200_cfi',Kby(10,100))
steps['SingleElectronPt10_UPG2017']=gen2017('SingleElectronPt10_cfi',Kby(9,300))
steps['SingleElectronPt35_UPG2017']=gen2017('SingleElectronPt35_cfi',Kby(9,500))
steps['SingleElectronPt1000_UPG2017']=gen2017('SingleElectronPt1000_cfi',Kby(9,50))
steps['SingleGammaPt10_UPG2017']=gen2017('SingleGammaPt10_cfi',Kby(9,300))
steps['SingleGammaPt35_UPG2017']=gen2017('SingleGammaPt35_cfi',Kby(9,50))
steps['SingleMuPt1_UPG2017']=gen2017('SingleMuPt1_cfi',Kby(25,1000))
steps['SingleMuPt10_UPG2017']=gen2017('SingleMuPt10_cfi',Kby(25,500))
steps['SingleMuPt100_UPG2017']=gen2017('SingleMuPt100_cfi',Kby(9,500))
steps['SingleMuPt1000_UPG2017']=gen2017('SingleMuPt1000_cfi',Kby(9,500))

steps['TTbarLepton_UPG2017_8']=gen2017('TTbarLepton_Tauola_8TeV_cfi',Kby(9,100))
steps['Wjet_Pt_80_120_UPG2017_8']=gen2017('Wjet_Pt_80_120_8TeV_cfi',Kby(9,100))
steps['Wjet_Pt_3000_3500_UPG2017_8']=gen2017('Wjet_Pt_3000_3500_8TeV_cfi',Kby(9,50))
steps['LM1_sfts_UPG2017_8']=gen2017('LM1_sfts_8TeV_cfi',Kby(9,100))

steps['QCD_Pt_3000_3500_UPG2017_8']=gen2017('QCD_Pt_3000_3500_8TeV_cfi',Kby(9,25))
steps['QCD_Pt_600_800_UPG2017_8']=gen2017('QCD_Pt_600_800_8TeV_cfi',Kby(9,50))
steps['QCD_Pt_80_120_UPG2017_8']=gen2017('QCD_Pt_80_120_8TeV_cfi',Kby(9,100))

steps['Higgs200ChargedTaus_UPG2017_8']=gen2017('H200ChargedTaus_Tauola_8TeV_cfi',Kby(9,100))
steps['JpsiMM_UPG2017_8']=gen2017('JpsiMM_8TeV_cfi',Kby(66,1000))
steps['TTbar_UPG2017_8']=gen2017('TTbar_Tauola_8TeV_cfi',Kby(9,100))
steps['WE_UPG2017_8']=gen2017('WE_8TeV_cfi',Kby(9,100))
steps['ZEE_UPG2017_8']=gen2017('ZEE_8TeV_cfi',Kby(9,100))
steps['ZTT_UPG2017_8']=gen2017('ZTT_Tauola_All_hadronic_8TeV_cfi',Kby(9,15))
steps['H130GGgluonfusion_UPG2017_8']=gen2017('H130GGgluonfusion_8TeV_cfi',Kby(9,100))
steps['PhotonJets_Pt_10_UPG2017_8']=gen2017('PhotonJet_Pt_10_8TeV_cfi',Kby(9,150))
steps['QQH1352T_Tauola_UPG2017_8']=gen2017('QQH1352T_Tauola_8TeV_cfi',Kby(9,100))

steps['MinBias_TuneZ2star_UPG2017_8']=gen2017('MinBias_TuneZ2star_8TeV_pythia6_cff',Kby(9,30))
steps['WM_UPG2017_8']=gen2017('WM_8TeV_cfi',Kby(9,200))
steps['ZMM_UPG2017_8']=gen2017('ZMM_8TeV_cfi',Kby(18,300))

steps['ADDMonoJet_d3MD3_UPG2017_8']=gen2017('ADDMonoJet_8TeV_d3MD3_cfi',Kby(9,100))
steps['ZpMM_UPG2017_8']=gen2017('ZpMM_8TeV_cfi',Kby(9,200))
steps['WpM_UPG2017_8']=gen2017('WpM_8TeV_cfi',Kby(9,200))



#14TeV
#steps['TTbarLepton_UPG2017_14']=gen2017('TTbarLepton_Tauola_14TeV_cfi',Kby(9,100))
steps['Wjet_Pt_80_120_UPG2017_14']=gen2017('Wjet_Pt_80_120_14TeV_cfi',Kby(9,100))
steps['Wjet_Pt_3000_3500_UPG2017_14']=gen2017('Wjet_Pt_3000_3500_14TeV_cfi',Kby(9,50))
steps['LM1_sfts_UPG2017_14']=gen2017('LM1_sfts_14TeV_cfi',Kby(9,100))

steps['QCD_Pt_3000_3500_UPG2017_14']=gen2017('QCD_Pt_3000_3500_14TeV_cfi',Kby(9,25))
#steps['QCD_Pt_600_800_UPG2017_14']=gen2017('QCD_Pt_600_800_14TeV_cfi',Kby(9,50))
steps['QCD_Pt_80_120_UPG2017_14']=gen2017('QCD_Pt_80_120_14TeV_cfi',Kby(9,100))

steps['Higgs200ChargedTaus_UPG2017_14']=gen2017('H200ChargedTaus_Tauola_14TeV_cfi',Kby(9,100))
steps['JpsiMM_UPG2017_14']=gen2017('JpsiMM_14TeV_cfi',Kby(66,1000))
steps['TTbar_UPG2017_14']=gen2017('TTbar_Tauola_14TeV_cfi',Kby(9,100))
steps['WE_UPG2017_14']=gen2017('WE_14TeV_cfi',Kby(9,100))
steps['ZEE_UPG2017_14']=gen2017('ZEE_14TeV_cfi',Kby(9,100))
steps['ZTT_UPG2017_14']=gen2017('ZTT_Tauola_All_hadronic_14TeV_cfi',Kby(9,150))
steps['H130GGgluonfusion_UPG2017_14']=gen2017('H130GGgluonfusion_14TeV_cfi',Kby(9,100))
steps['PhotonJets_Pt_10_UPG2017_14']=gen2017('PhotonJet_Pt_10_14TeV_cfi',Kby(9,150))
steps['QQH1352T_Tauola_UPG2017_14']=gen2017('QQH1352T_Tauola_14TeV_cfi',Kby(9,100))

steps['MinBias_TuneZ2star_UPG2017_14']=gen2017('MinBias_TuneZ2star_14TeV_pythia6_cff',Kby(9,300))
steps['WM_UPG2017_14']=gen2017('WM_14TeV_cfi',Kby(9,200))
steps['ZMM_UPG2017_14']=gen2017('ZMM_14TeV_cfi',Kby(18,300))

#steps['ADDMonoJet_d3MD3_UPG2017_14']=gen2017('ADDMonoJet_14TeV_d3MD3_cfi',Kby(9,100))
#steps['ZpMM_UPG2017_14']=gen2017('ZpMM_14TeV_cfi',Kby(9,200))
#steps['WpM_UPG2017_14']=gen2017('WpM_14TeV_cfi',Kby(9,200))

# step1 gensim
step1Up2017EcalFineDefaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'auto:upgrade2017', 
                             '--beamspot' : 'Gauss',
			     '--magField' : '38T_PostLS1',
			     '--datatier' : 'GEN-SIM',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'Extended2017',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/customiseECalSD_1ps_granularity'
                             }
def gen2017EcalFine(fragment,howMuch):
    global step1Up2017EcalFineDefaults
    return merge([{'cfg':fragment},howMuch,step1Up2017EcalFineDefaults])


steps['FourMuPt1_200_UPG2017EcalFine']=gen2017EcalFine('FourMuPt_1_200_cfi',Kby(10,100))
steps['SingleElectronPt10_UPG2017EcalFine']=gen2017EcalFine('SingleElectronPt10_cfi',Kby(9,300))
steps['SingleElectronPt35_UPG2017EcalFine']=gen2017EcalFine('SingleElectronPt35_cfi',Kby(9,500))
steps['SingleElectronPt1000_UPG2017EcalFine']=gen2017EcalFine('SingleElectronPt1000_cfi',Kby(9,50))
steps['SingleGammaPt10_UPG2017EcalFine']=gen2017EcalFine('SingleGammaPt10_cfi',Kby(9,300))
steps['SingleGammaPt35_UPG2017EcalFine']=gen2017EcalFine('SingleGammaPt35_cfi',Kby(9,50))
steps['SingleMuPt1_UPG2017EcalFine']=gen2017EcalFine('SingleMuPt1_cfi',Kby(25,1000))
steps['SingleMuPt10_UPG2017EcalFine']=gen2017EcalFine('SingleMuPt10_cfi',Kby(25,500))
steps['SingleMuPt100_UPG2017EcalFine']=gen2017EcalFine('SingleMuPt100_cfi',Kby(9,500))
steps['SingleMuPt1000_UPG2017EcalFine']=gen2017EcalFine('SingleMuPt1000_cfi',Kby(9,500))

steps['TTbarLepton_UPG2017EcalFine_8']=gen2017EcalFine('TTbarLepton_Tauola_8TeV_cfi',Kby(9,100))
steps['Wjet_Pt_80_120_UPG2017EcalFine_8']=gen2017EcalFine('Wjet_Pt_80_120_8TeV_cfi',Kby(9,100))
steps['Wjet_Pt_3000_3500_UPG2017EcalFine_8']=gen2017EcalFine('Wjet_Pt_3000_3500_8TeV_cfi',Kby(9,50))
steps['LM1_sfts_UPG2017EcalFine_8']=gen2017EcalFine('LM1_sfts_8TeV_cfi',Kby(9,100))

steps['QCD_Pt_3000_3500_UPG2017EcalFine_8']=gen2017EcalFine('QCD_Pt_3000_3500_8TeV_cfi',Kby(9,25))
steps['QCD_Pt_600_800_UPG2017EcalFine_8']=gen2017EcalFine('QCD_Pt_600_800_8TeV_cfi',Kby(9,50))
steps['QCD_Pt_80_120_UPG2017EcalFine_8']=gen2017EcalFine('QCD_Pt_80_120_8TeV_cfi',Kby(9,100))

steps['Higgs200ChargedTaus_UPG2017EcalFine_8']=gen2017EcalFine('H200ChargedTaus_Tauola_8TeV_cfi',Kby(9,100))
steps['JpsiMM_UPG2017EcalFine_8']=gen2017EcalFine('JpsiMM_8TeV_cfi',Kby(66,1000))
steps['TTbar_UPG2017EcalFine_8']=gen2017EcalFine('TTbar_Tauola_8TeV_cfi',Kby(9,100))
steps['WE_UPG2017EcalFine_8']=gen2017EcalFine('WE_8TeV_cfi',Kby(9,100))
steps['ZEE_UPG2017EcalFine_8']=gen2017EcalFine('ZEE_8TeV_cfi',Kby(9,100))
steps['ZTT_UPG2017EcalFine_8']=gen2017EcalFine('ZTT_Tauola_All_hadronic_8TeV_cfi',Kby(9,15))
steps['H130GGgluonfusion_UPG2017EcalFine_8']=gen2017EcalFine('H130GGgluonfusion_8TeV_cfi',Kby(9,100))
steps['PhotonJets_Pt_10_UPG2017EcalFine_8']=gen2017EcalFine('PhotonJet_Pt_10_8TeV_cfi',Kby(9,150))
steps['QQH1352T_Tauola_UPG2017EcalFine_8']=gen2017EcalFine('QQH1352T_Tauola_8TeV_cfi',Kby(9,100))

steps['MinBias_TuneZ2star_UPG2017EcalFine_8']=gen2017EcalFine('MinBias_TuneZ2star_8TeV_pythia6_cff',Kby(9,30))
steps['WM_UPG2017EcalFine_8']=gen2017EcalFine('WM_8TeV_cfi',Kby(9,200))
steps['ZMM_UPG2017EcalFine_8']=gen2017EcalFine('ZMM_8TeV_cfi',Kby(18,300))

steps['ADDMonoJet_d3MD3_UPG2017EcalFine_8']=gen2017EcalFine('ADDMonoJet_8TeV_d3MD3_cfi',Kby(9,100))
steps['ZpMM_UPG2017EcalFine_8']=gen2017EcalFine('ZpMM_8TeV_cfi',Kby(9,200))
steps['WpM_UPG2017EcalFine_8']=gen2017EcalFine('WpM_8TeV_cfi',Kby(9,200))



#14TeV
steps['TTbarLepton_UPG2017EcalFine_14']=gen2017EcalFine('TTbarLepton_Tauola_14TeV_cfi',Kby(9,100))
steps['Wjet_Pt_80_120_UPG2017EcalFine_14']=gen2017EcalFine('Wjet_Pt_80_120_14TeV_cfi',Kby(9,100))
steps['Wjet_Pt_3000_3500_UPG2017EcalFine_14']=gen2017EcalFine('Wjet_Pt_3000_3500_14TeV_cfi',Kby(9,50))
steps['LM1_sfts_UPG2017EcalFine_14']=gen2017EcalFine('LM1_sfts_14TeV_cfi',Kby(9,100))

steps['QCD_Pt_3000_3500_UPG2017EcalFine_14']=gen2017EcalFine('QCD_Pt_3000_3500_14TeV_cfi',Kby(9,25))
steps['QCD_Pt_600_800_UPG2017EcalFine_14']=gen2017EcalFine('QCD_Pt_600_800_14TeV_cfi',Kby(9,50))
steps['QCD_Pt_80_120_UPG2017EcalFine_14']=gen2017EcalFine('QCD_Pt_80_120_14TeV_cfi',Kby(9,100))

steps['Higgs200ChargedTaus_UPG2017EcalFine_14']=gen2017EcalFine('H200ChargedTaus_Tauola_14TeV_cfi',Kby(9,100))
steps['JpsiMM_UPG2017EcalFine_14']=gen2017EcalFine('JpsiMM_14TeV_cfi',Kby(66,1000))
steps['TTbar_UPG2017EcalFine_14']=gen2017EcalFine('TTbar_Tauola_14TeV_cfi',Kby(9,100))
steps['WE_UPG2017EcalFine_14']=gen2017EcalFine('WE_14TeV_cfi',Kby(9,100))
steps['ZEE_UPG2017EcalFine_14']=gen2017EcalFine('ZEE_14TeV_cfi',Kby(9,100))
steps['ZTT_UPG2017EcalFine_14']=gen2017EcalFine('ZTT_Tauola_All_hadronic_14TeV_cfi',Kby(9,150))
steps['H130GGgluonfusion_UPG2017EcalFine_14']=gen2017EcalFine('H130GGgluonfusion_14TeV_cfi',Kby(9,100))
steps['PhotonJets_Pt_10_UPG2017EcalFine_14']=gen2017EcalFine('PhotonJet_Pt_10_14TeV_cfi',Kby(9,150))
steps['QQH1352T_Tauola_UPG2017EcalFine_14']=gen2017EcalFine('QQH1352T_Tauola_14TeV_cfi',Kby(9,100))

steps['MinBias_TuneZ2star_UPG2017EcalFine_14']=gen2017EcalFine('MinBias_TuneZ2star_14TeV_pythia6_cff',Kby(9,300))
steps['WM_UPG2017EcalFine_14']=gen2017EcalFine('WM_14TeV_cfi',Kby(9,200))
steps['ZMM_UPG2017EcalFine_14']=gen2017EcalFine('ZMM_14TeV_cfi',Kby(18,300))

steps['ADDMonoJet_d3MD3_UPG2017EcalFine_14']=gen2017EcalFine('ADDMonoJet_14TeV_d3MD3_cfi',Kby(9,100))
steps['ZpMM_UPG2017EcalFine_14']=gen2017EcalFine('ZpMM_14TeV_cfi',Kby(9,200))
steps['WpM_UPG2017EcalFine_14']=gen2017EcalFine('WpM_14TeV_cfi',Kby(9,200))



####GENSIM AGING VALIDATION - STARTUP set of reference

step1Up2017_START_Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'W17_150_62E2::All', 
                             '--beamspot' : 'Gauss',
                             '--magField' : '38T_PostLS1',
                             '--datatier' : 'GEN-SIM',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'Extended2017',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017'
                             }
def gen2017start(fragment,howMuch):
    global step1Up2017_START_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2017_START_Defaults])


####GENSIM AGING VALIDATION - 300fb-1

step1Up2017_300_Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'W17_300_62E2::All', 
                             '--beamspot' : 'Gauss',
                             '--datatier' : 'GEN-SIM',
                             '--magField' : '38T_PostLS1',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'Extended2017',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_300'
                             }
def gen2017300(fragment,howMuch):
    global step1Up2017_300_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2017_300_Defaults])


####GENSIM AGING VALIDATION - 300fb-1 COMPLETE ECAL

step1Up2017_300comp_Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'W17_300_62C2::All', 
                             '--beamspot' : 'Gauss',
                             '--datatier' : 'GEN-SIM',
                             '--magField' : '38T_PostLS1',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'Extended2017',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_300,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_300'
                             }
def gen2017300comp(fragment,howMuch):
    global step1Up2017_300comp_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2017_300comp_Defaults])

####GENSIM AGING VALIDATION - 500fb-1

step1Up2017_500_Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'W17_500_62E2::All', 
                             '--beamspot' : 'Gauss',
                             '--magField' : '38T_PostLS1',
                             '--datatier' : 'GEN-SIM',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'Extended2017',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_500'
                             }
def gen2017500(fragment,howMuch):
    global step1Up2017_500_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2017_500_Defaults])


####GENSIM AGING VALIDATION - 1000fb-1

step1Up2017_1000_Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'W17_100062E2::All', 
                             '--beamspot' : 'Gauss',
                             '--magField' : '38T_PostLS1',
                             '--datatier' : 'GEN-SIM',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'Extended2017',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000'
                             }
def gen20171000(fragment,howMuch):
    global step1Up2017_1000_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2017_1000_Defaults])

####GENSIM AGING VALIDATION - 1000fb-1 COMPLETE ECAL

step1Up2017_1000comp_Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'W17_100062C2::All', 
                             '--beamspot' : 'Gauss',
                              '--magField' : '38T_PostLS1',
                             '--datatier' : 'GEN-SIM',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'Extended2017',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_1000'
                             }
def gen20171000comp(fragment,howMuch):
    global step1Up2017_1000comp_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2017_1000comp_Defaults])


####GENSIM AGING VALIDATION - 1000fb-1 TkId

step1Up2017_1000_TkId_Defaults = {'-s' : 'GEN,SIM',
                                     '-n' : 10,
                                     '--conditions' : 'W17_100062E2A::All', 
                                     '--beamspot' : 'Gauss',
                                     '--datatier' : 'GEN-SIM',
                                     '--magField' : '38T_PostLS1',
                                     '--eventcontent': 'FEVTDEBUG',
                                     '--geometry' : 'Extended2017',
                                     '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000'
                             }
def gen20171000TkId(fragment,howMuch):
    global step1Up2017_1000_TkId_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2017_1000_TkId_Defaults])

####GENSIM AGING VALIDATION - 1000fb-1 TkId COMPLETE ECAL

step1Up2017_1000comp_TkId_Defaults = {'-s' : 'GEN,SIM',
                                     '-n' : 10,
                                     '--conditions' : 'W17_100062C2A::All', 
                                     '--beamspot' : 'Gauss',
                                     '--datatier' : 'GEN-SIM',
                                     '--magField' : '38T_PostLS1',
                                     '--eventcontent': 'FEVTDEBUG',
                                     '--geometry' : 'Extended2017',
                                     '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_1000'
                             }
def gen20171000compTkId(fragment,howMuch):
    global step1Up2017_1000comp_TkId_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2017_1000comp_TkId_Defaults])

####GENSIM AGING VALIDATION - 3000fb-1 

step1Up2017_3000_Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'W17_300062E2::All', 
                             '--beamspot' : 'Gauss',
                             '--datatier' : 'GEN-SIM',
                             '--magField' : '38T_PostLS1',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'Extended2017',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_3000'
                             }
def gen20173000(fragment,howMuch):
    global step1Up2017_3000_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2017_3000_Defaults])


####GENSIM AGING VALIDATION - 3000fb-1 COMPLETE ECAL

step1Up2017_3000comp_Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'W17_300062C2::All', 
                             '--beamspot' : 'Gauss',
                             '--magField' : '38T_PostLS1',
                             '--datatier' : 'GEN-SIM',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'Extended2017',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_3000,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_3000'
                             }
def gen20173000comp(fragment,howMuch):
    global step1Up2017_3000comp_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2017_3000comp_Defaults])

steps['FourMuPt1_200_UPG2017_STAR']=gen2017start('FourMuPt_1_200_cfi',Kby(10,100))
steps['FourMuPt1_200_UPG2017_300']=gen2017300('FourMuPt_1_200_cfi',Kby(10,100))
steps['FourMuPt1_200_UPG2017_500']=gen2017500('FourMuPt_1_200_cfi',Kby(10,100))
steps['FourMuPt1_200_UPG2017_1000']=gen20171000('FourMuPt_1_200_cfi',Kby(10,100))
steps['FourMuPt1_200_UPG2017_1000TkId']=gen20171000TkId('FourMuPt_1_200_cfi',Kby(10,100))
steps['FourMuPt1_200_UPG2017_3000']=gen20173000('FourMuPt_1_200_cfi',Kby(10,100))

steps['FourMuPt1_200_UPG2017PU20_STAR']=gen2017start('FourMuPt_1_200_cfi',Kby(10,100))
steps['FourMuPt1_200_UPG2017PU20_300']=gen2017300('FourMuPt_1_200_cfi',Kby(10,100))
steps['FourMuPt1_200_UPG2017PU20_500']=gen2017500('FourMuPt_1_200_cfi',Kby(10,100))
steps['FourMuPt1_200_UPG2017PU20_1000']=gen20171000('FourMuPt_1_200_cfi',Kby(10,100))

steps['TenMuE_0_200_UPG2017_STAR']=gen2017start('TenMuE_0_200_cfi',Kby(10,100))
steps['TenMuE_0_200_UPG2017_300']=gen2017300('TenMuE_0_200_cfi',Kby(10,100))
steps['TenMuE_0_200_UPG2017_500']=gen2017500('TenMuE_0_200_cfi',Kby(10,100))
steps['TenMuE_0_200_UPG2017_1000']=gen20171000('TenMuE_0_200_cfi',Kby(10,100))
steps['TenMuE_0_200_UPG2017_1000TkId']=gen20171000TkId('TenMuE_0_200_cfi',Kby(10,100))
steps['TenMuE_0_200_UPG2017_3000']=gen20173000('TenMuE_0_200_cfi',Kby(10,100))

steps['TenMuE_0_200_UPG2017PU20_STAR']=gen2017start('TenMuE_0_200_cfi',Kby(10,100))
steps['TenMuE_0_200_UPG2017PU20_300']=gen2017300('TenMuE_0_200_cfi',Kby(10,100))
steps['TenMuE_0_200_UPG2017PU20_500']=gen2017500('TenMuE_0_200_cfi',Kby(10,100))
steps['TenMuE_0_200_UPG2017PU20_1000']=gen20171000('TenMuE_0_200_cfi',Kby(10,100))

steps['MinBias_TuneZ2star_UPG2017_14_STAR']=gen2017start('MinBias_TuneZ2star_14TeV_pythia6_cff',Kby(9,300))
steps['MinBias_TuneZ2star_UPG2017_14_300']=gen2017300('MinBias_TuneZ2star_14TeV_pythia6_cff',Kby(9,300))
steps['MinBias_TuneZ2star_UPG2017_14_500']=gen2017500('MinBias_TuneZ2star_14TeV_pythia6_cff',Kby(9,300))
steps['MinBias_TuneZ2star_UPG2017_14_1000']=gen20171000('MinBias_TuneZ2star_14TeV_pythia6_cff',Kby(9,300))
steps['MinBias_TuneZ2star_UPG2017_14_1000TkId']=gen20171000TkId('MinBias_TuneZ2star_14TeV_pythia6_cff',Kby(9,300))
steps['MinBias_TuneZ2star_UPG2017_14_3000']=gen20173000('MinBias_TuneZ2star_14TeV_pythia6_cff',Kby(9,300))

steps['ZEE_UPG2017_14_STAR']=gen2017start('ZEE_14TeV_cfi',Kby(9,100))
steps['ZEE_UPG2017_14_300']=gen2017300('ZEE_14TeV_cfi',Kby(9,100))
steps['ZEE_UPG2017_14_300COMP']=gen2017300comp('ZEE_14TeV_cfi',Kby(9,100))
steps['ZEE_UPG2017_14_500']=gen2017500('ZEE_14TeV_cfi',Kby(9,100))
steps['ZEE_UPG2017_14_1000']=gen20171000('ZEE_14TeV_cfi',Kby(9,100))
steps['ZEE_UPG2017_14_1000COMP']=gen20171000comp('ZEE_14TeV_cfi',Kby(9,100))
steps['ZEE_UPG2017_14_1000TkId']=gen20171000TkId('ZEE_14TeV_cfi',Kby(9,100))
steps['ZEE_UPG2017_14_1000COMPTkId']=gen20171000compTkId('ZEE_14TeV_cfi',Kby(9,100))
steps['ZEE_UPG2017_14_3000']=gen20173000('ZEE_14TeV_cfi',Kby(9,100))
steps['ZEE_UPG2017_14_3000COMP']=gen20173000comp('ZEE_14TeV_cfi',Kby(9,100))

steps['TTbar_UPG2017_14_STAR']=gen2017start('TTbar_Tauola_14TeV_cfi',Kby(9,100))
steps['TTbar_UPG2017_14_300']=gen2017300('TTbar_Tauola_14TeV_cfi',Kby(9,100))
steps['TTbar_UPG2017_14_500']=gen2017500('TTbar_Tauola_14TeV_cfi',Kby(9,100))
steps['TTbar_UPG2017_14_1000']=gen20171000('TTbar_Tauola_14TeV_cfi',Kby(9,100))
steps['TTbar_UPG2017_14_1000TkId']=gen20171000TkId('TTbar_Tauola_14TeV_cfi',Kby(9,100))
steps['TTbar_UPG2017_14_3000']=gen20173000('TTbar_Tauola_14TeV_cfi',Kby(9,100))

steps['TTbar_UPG2017PU20_14_STAR']=gen2017start('TTbar_Tauola_14TeV_cfi',Kby(9,100))
steps['TTbar_UPG2017PU20_14_300']=gen2017300('TTbar_Tauola_14TeV_cfi',Kby(9,100))
steps['TTbar_UPG2017PU20_14_500']=gen2017500('TTbar_Tauola_14TeV_cfi',Kby(9,100))
steps['TTbar_UPG2017PU20_14_1000']=gen20171000('TTbar_Tauola_14TeV_cfi',Kby(9,100))

## 2019

step1Up2019Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'auto:upgrade2019', 
                             '--beamspot' : 'Gauss',
                             '--datatier' : 'GEN-SIM',
                             '--magField' : '38T_PostLS1',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'Extended2019',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019'
                             }
def gen2019(fragment,howMuch):
    global step1Up2019Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2019Defaults])

steps['FourMuPt1_200_UPG2019']=gen2019('FourMuPt_1_200_cfi',Kby(10,100))
steps['SingleElectronPt10_UPG2019']=gen2019('SingleElectronPt10_cfi',Kby(9,300))
steps['SingleElectronPt35_UPG2019']=gen2019('SingleElectronPt35_cfi',Kby(9,500))
steps['SingleElectronPt1000_UPG2019']=gen2019('SingleElectronPt1000_cfi',Kby(9,50))
steps['SingleGammaPt10_UPG2019']=gen2019('SingleGammaPt10_cfi',Kby(9,300))
steps['SingleGammaPt35_UPG2019']=gen2019('SingleGammaPt35_cfi',Kby(9,50))
steps['SingleMuPt1_UPG2019']=gen2019('SingleMuPt1_cfi',Kby(25,1000))
steps['SingleMuPt10_UPG2019']=gen2019('SingleMuPt10_cfi',Kby(25,500))
steps['SingleMuPt100_UPG2019']=gen2019('SingleMuPt100_cfi',Kby(9,500))
steps['SingleMuPt1000_UPG2019']=gen2019('SingleMuPt1000_cfi',Kby(9,500))

steps['TTbarLepton_UPG2019_8']=gen2019('TTbarLepton_Tauola_8TeV_cfi',Kby(9,100))
steps['Wjet_Pt_80_120_UPG2019_8']=gen2019('Wjet_Pt_80_120_8TeV_cfi',Kby(9,100))
steps['Wjet_Pt_3000_3500_UPG2019_8']=gen2019('Wjet_Pt_3000_3500_8TeV_cfi',Kby(9,50))
steps['LM1_sfts_UPG2019_8']=gen2019('LM1_sfts_8TeV_cfi',Kby(9,100))

steps['QCD_Pt_3000_3500_UPG2019_8']=gen2019('QCD_Pt_3000_3500_8TeV_cfi',Kby(9,25))
steps['QCD_Pt_600_800_UPG2019_8']=gen2019('QCD_Pt_600_800_8TeV_cfi',Kby(9,50))
steps['QCD_Pt_80_120_UPG2019_8']=gen2019('QCD_Pt_80_120_8TeV_cfi',Kby(9,100))

steps['Higgs200ChargedTaus_UPG2019_8']=gen2019('H200ChargedTaus_Tauola_8TeV_cfi',Kby(9,100))
steps['JpsiMM_UPG2019_8']=gen2019('JpsiMM_8TeV_cfi',Kby(66,1000))
steps['TTbar_UPG2019_8']=gen2019('TTbar_Tauola_8TeV_cfi',Kby(9,100))
steps['WE_UPG2019_8']=gen2019('WE_8TeV_cfi',Kby(9,100))
steps['ZEE_UPG2019_8']=gen2019('ZEE_8TeV_cfi',Kby(9,100))
steps['ZTT_UPG2019_8']=gen2019('ZTT_Tauola_All_hadronic_8TeV_cfi',Kby(9,150))
steps['H130GGgluonfusion_UPG2019_8']=gen2019('H130GGgluonfusion_8TeV_cfi',Kby(9,100))
steps['PhotonJets_Pt_10_UPG2019_8']=gen2019('PhotonJet_Pt_10_8TeV_cfi',Kby(9,150))
steps['QQH1352T_Tauola_UPG2019_8']=gen2019('QQH1352T_Tauola_8TeV_cfi',Kby(9,100))

steps['MinBias_TuneZ2star_UPG2019_8']=gen2019('MinBias_TuneZ2star_8TeV_pythia6_cff',Kby(9,300))
steps['WM_UPG2019_8']=gen2019('WM_8TeV_cfi',Kby(9,200))
steps['ZMM_UPG2019_8']=gen2019('ZMM_8TeV_cfi',Kby(18,300))

steps['ADDMonoJet_d3MD3_UPG2019_8']=gen2019('ADDMonoJet_8TeV_d3MD3_cfi',Kby(9,100))
steps['ZpMM_UPG2019_8']=gen2019('ZpMM_8TeV_cfi',Kby(9,200))
steps['WpM_UPG2019_8']=gen2019('WpM_8TeV_cfi',Kby(9,200))


#14TeV
#steps['TTbarLepton_UPG2019_14']=gen2019('TTbarLepton_Tauola_14TeV_cfi',Kby(9,100))
steps['Wjet_Pt_80_120_UPG2019_14']=gen2019('Wjet_Pt_80_120_14TeV_cfi',Kby(9,100))
steps['Wjet_Pt_3000_3500_UPG2019_14']=gen2019('Wjet_Pt_3000_3500_14TeV_cfi',Kby(9,50))
steps['LM1_sfts_UPG2019_14']=gen2019('LM1_sfts_14TeV_cfi',Kby(9,100))

steps['QCD_Pt_3000_3500_UPG2019_14']=gen2019('QCD_Pt_3000_3500_14TeV_cfi',Kby(9,25))
#steps['QCD_Pt_600_800_UPG2019_14']=gen2019('QCD_Pt_600_800_14TeV_cfi',Kby(9,50))
steps['QCD_Pt_80_120_UPG2019_14']=gen2019('QCD_Pt_80_120_14TeV_cfi',Kby(9,100))

steps['Higgs200ChargedTaus_UPG2019_14']=gen2019('H200ChargedTaus_Tauola_14TeV_cfi',Kby(9,100))
steps['JpsiMM_UPG2019_14']=gen2019('JpsiMM_14TeV_cfi',Kby(66,1000))
steps['TTbar_UPG2019_14']=gen2019('TTbar_Tauola_14TeV_cfi',Kby(9,100))
steps['WE_UPG2019_14']=gen2019('WE_14TeV_cfi',Kby(9,100))
steps['ZEE_UPG2019_14']=gen2019('ZEE_14TeV_cfi',Kby(9,100))
steps['ZTT_UPG2019_14']=gen2019('ZTT_Tauola_All_hadronic_14TeV_cfi',Kby(9,150))
steps['H130GGgluonfusion_UPG2019_14']=gen2019('H130GGgluonfusion_14TeV_cfi',Kby(9,100))
steps['PhotonJets_Pt_10_UPG2019_14']=gen2019('PhotonJet_Pt_10_14TeV_cfi',Kby(9,150))
steps['QQH1352T_Tauola_UPG2019_14']=gen2019('QQH1352T_Tauola_14TeV_cfi',Kby(9,100))

steps['MinBias_TuneZ2star_UPG2019_14']=gen2019('MinBias_TuneZ2star_14TeV_pythia6_cff',Kby(9,300))
steps['WM_UPG2019_14']=gen2019('WM_14TeV_cfi',Kby(9,200))
steps['ZMM_UPG2019_14']=gen2019('ZMM_14TeV_cfi',Kby(18,300))

#steps['ADDMonoJet_d3MD3_UPG2019_14']=gen2019('ADDMonoJet_14TeV_d3MD3_cfi',Kby(9,100))
#steps['ZpMM_UPG2019_14']=gen2019('ZpMM_14TeV_cfi',Kby(9,200))
#steps['WpM_UPG2019_14']=gen2019('WpM_14TeV_cfi',Kby(9,200))


####GENSIM AGING VALIDATION - STARTUP set of reference

step1Up2019_START_Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'W19_150_62E2::All', 
                             '--beamspot' : 'Gauss',
                             '--datatier' : 'GEN-SIM',
                             '--magField' : '38T_PostLS1',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'Extended2019',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019'
                             }
def gen2019start(fragment,howMuch):
    global step1Up2019_START_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2019_START_Defaults])







####GENSIM AGING VALIDATION - 300fb-1

step1Up2019_300_Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'W19_300_62E2::All', 
                             '--beamspot' : 'Gauss',
                             '--datatier' : 'GEN-SIM',
                             '--magField' : '38T_PostLS1',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'Extended2019',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_300'
                             }
def gen2019300(fragment,howMuch):
    global step1Up2019_300_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2019_300_Defaults])


####GENSIM AGING VALIDATION - 300fb-1 COMPLETE ECAL

step1Up2019_300comp_Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'W19_300_62C2::All', 
                             '--beamspot' : 'Gauss',
                             '--datatier' : 'GEN-SIM',
                             '--magField' : '38T_PostLS1',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'Extended2019',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_300,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_300'
                             }
def gen2019300comp(fragment,howMuch):
    global step1Up2019_300comp_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2019_300comp_Defaults])

####GENSIM AGING VALIDATION - 500fb-1

step1Up2019_500_Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'W19_500_62E2::All', 
                             '--beamspot' : 'Gauss',
                             '--datatier' : 'GEN-SIM',
                             '--magField' : '38T_PostLS1',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'Extended2019',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_500'
                             }
def gen2019500(fragment,howMuch):
    global step1Up2019_500_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2019_500_Defaults])


####GENSIM AGING VALIDATION - 1000fb-1

step1Up2019_1000_Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'W19_100062E2::All', 
                             '--beamspot' : 'Gauss',
                             '--datatier' : 'GEN-SIM',
                             '--magField' : '38T_PostLS1',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'Extended2019',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000'
                             }
def gen20191000(fragment,howMuch):
    global step1Up2019_1000_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2019_1000_Defaults])

####GENSIM AGING VALIDATION - 1000fb-1 COMPLETE ECAL

step1Up2019_1000comp_Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'W19_100062C2::All', 
                             '--beamspot' : 'Gauss',
                             '--datatier' : 'GEN-SIM',
                             '--magField' : '38T_PostLS1',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'Extended2019',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_1000'
                             }
def gen20191000comp(fragment,howMuch):
    global step1Up2019_1000comp_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2019_1000comp_Defaults])


####GENSIM AGING VALIDATION - 1000fb-1 TkId

step1Up2019_1000_TkId_Defaults = {'-s' : 'GEN,SIM',
                                     '-n' : 10,
                                     '--conditions' : 'W19_100062E2A::All', 
                                     '--beamspot' : 'Gauss',
                                     '--datatier' : 'GEN-SIM',
                                     '--magField' : '38T_PostLS1',
                                     '--eventcontent': 'FEVTDEBUG',
                                     '--geometry' : 'Extended2019',
                                     '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000'
                             }
def gen20191000TkId(fragment,howMuch):
    global step1Up2019_1000_TkId_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2019_1000_TkId_Defaults])

####GENSIM AGING VALIDATION - 1000fb-1 TkId COMPLETE ECAL

step1Up2019_1000comp_TkId_Defaults = {'-s' : 'GEN,SIM',
                                     '-n' : 10,
                                     '--conditions' : 'W19_100062C2A::All', 
                                     '--beamspot' : 'Gauss',
                                     '--datatier' : 'GEN-SIM',
                                     '--magField' : '38T_PostLS1',
                                     '--eventcontent': 'FEVTDEBUG',
                                     '--geometry' : 'Extended2019',
                                     '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_1000'
                             }
def gen20191000compTkId(fragment,howMuch):
    global step1Up2019_1000comp_TkId_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2019_1000comp_TkId_Defaults])

####GENSIM AGING VALIDATION - 3000fb-1 

step1Up2019_3000_Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'W19_300062E2::All', 
                             '--beamspot' : 'Gauss',
                             '--datatier' : 'GEN-SIM',
                             '--magField' : '38T_PostLS1',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'Extended2019',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_3000'
                             }
def gen20193000(fragment,howMuch):
    global step1Up2019_3000_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2019_3000_Defaults])


####GENSIM AGING VALIDATION - 3000fb-1 COMPLETE ECAL

step1Up2019_3000comp_Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'W19_300062C2::All', 
                             '--beamspot' : 'Gauss',
                             '--datatier' : 'GEN-SIM',
                             '--magField' : '38T_PostLS1',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'Extended2019',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_3000,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_3000'
                             }
def gen20193000comp(fragment,howMuch):
    global step1Up2019_3000comp_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2019_3000comp_Defaults])

steps['FourMuPt1_200_UPG2019_STAR']=gen2019start('FourMuPt_1_200_cfi',Kby(10,100))
steps['FourMuPt1_200_UPG2019_300']=gen2019300('FourMuPt_1_200_cfi',Kby(10,100))
steps['FourMuPt1_200_UPG2019_500']=gen2019500('FourMuPt_1_200_cfi',Kby(10,100))
steps['FourMuPt1_200_UPG2019_1000']=gen20191000('FourMuPt_1_200_cfi',Kby(10,100))
steps['FourMuPt1_200_UPG2019_1000TkId']=gen20191000TkId('FourMuPt_1_200_cfi',Kby(10,100))
steps['FourMuPt1_200_UPG2019_3000']=gen20193000('FourMuPt_1_200_cfi',Kby(10,100))

steps['FourMuPt1_200_UPG2019PU20_STAR']=gen2019start('FourMuPt_1_200_cfi',Kby(10,100))
steps['FourMuPt1_200_UPG2019PU20_300']=gen2019300('FourMuPt_1_200_cfi',Kby(10,100))
steps['FourMuPt1_200_UPG2019PU20_500']=gen2019500('FourMuPt_1_200_cfi',Kby(10,100))
steps['FourMuPt1_200_UPG2019PU20_1000']=gen20191000('FourMuPt_1_200_cfi',Kby(10,100))

steps['TenMuE_0_200_UPG2019_STAR']=gen2019start('TenMuE_0_200_cfi',Kby(10,100))
steps['TenMuE_0_200_UPG2019_300']=gen2019300('TenMuE_0_200_cfi',Kby(10,100))
steps['TenMuE_0_200_UPG2019_500']=gen2019500('TenMuE_0_200_cfi',Kby(10,100))
steps['TenMuE_0_200_UPG2019_1000']=gen20191000('TenMuE_0_200_cfi',Kby(10,100))
steps['TenMuE_0_200_UPG2019_1000TkId']=gen20191000TkId('TenMuE_0_200_cfi',Kby(10,100))
steps['TenMuE_0_200_UPG2019_3000']=gen20193000('TenMuE_0_200_cfi',Kby(10,100))

steps['TenMuE_0_200_UPG2019PU20_STAR']=gen2019start('TenMuE_0_200_cfi',Kby(10,100))
steps['TenMuE_0_200_UPG2019PU20_300']=gen2019300('TenMuE_0_200_cfi',Kby(10,100))
steps['TenMuE_0_200_UPG2019PU20_500']=gen2019500('TenMuE_0_200_cfi',Kby(10,100))
steps['TenMuE_0_200_UPG2019PU20_1000']=gen20191000('TenMuE_0_200_cfi',Kby(10,100))

steps['MinBias_TuneZ2star_UPG2019_14_STAR']=gen2019start('MinBias_TuneZ2star_14TeV_pythia6_cff',Kby(9,300))
steps['MinBias_TuneZ2star_UPG2019_14_300']=gen2019300('MinBias_TuneZ2star_14TeV_pythia6_cff',Kby(9,300))
steps['MinBias_TuneZ2star_UPG2019_14_500']=gen2019500('MinBias_TuneZ2star_14TeV_pythia6_cff',Kby(9,300))
steps['MinBias_TuneZ2star_UPG2019_14_1000']=gen20191000('MinBias_TuneZ2star_14TeV_pythia6_cff',Kby(9,300))
steps['MinBias_TuneZ2star_UPG2019_14_1000TkId']=gen20191000TkId('MinBias_TuneZ2star_14TeV_pythia6_cff',Kby(9,300))
steps['MinBias_TuneZ2star_UPG2019_14_3000']=gen20193000('MinBias_TuneZ2star_14TeV_pythia6_cff',Kby(9,300))

steps['ZEE_UPG2019_14_STAR']=gen2019start('ZEE_14TeV_cfi',Kby(9,100))
steps['ZEE_UPG2019_14_300']=gen2019300('ZEE_14TeV_cfi',Kby(9,100))
steps['ZEE_UPG2019_14_300COMP']=gen2019300comp('ZEE_14TeV_cfi',Kby(9,100))
steps['ZEE_UPG2019_14_500']=gen2019500('ZEE_14TeV_cfi',Kby(9,100))
steps['ZEE_UPG2019_14_1000']=gen20191000('ZEE_14TeV_cfi',Kby(9,100))
steps['ZEE_UPG2019_14_1000COMP']=gen20191000comp('ZEE_14TeV_cfi',Kby(9,100))
steps['ZEE_UPG2019_14_1000TkId']=gen20191000TkId('ZEE_14TeV_cfi',Kby(9,100))
steps['ZEE_UPG2019_14_1000COMPTkId']=gen20191000compTkId('ZEE_14TeV_cfi',Kby(9,100))
steps['ZEE_UPG2019_14_3000']=gen20193000('ZEE_14TeV_cfi',Kby(9,100))
steps['ZEE_UPG2019_14_3000COMP']=gen20193000comp('ZEE_14TeV_cfi',Kby(9,100))

steps['TTbar_UPG2019_14_STAR']=gen2019start('TTbar_Tauola_14TeV_cfi',Kby(9,100))
steps['TTbar_UPG2019_14_300']=gen2019300('TTbar_Tauola_14TeV_cfi',Kby(9,100))
steps['TTbar_UPG2019_14_500']=gen2019500('TTbar_Tauola_14TeV_cfi',Kby(9,100))
steps['TTbar_UPG2019_14_1000']=gen20191000('TTbar_Tauola_14TeV_cfi',Kby(9,100))
steps['TTbar_UPG2019_14_1000TkId']=gen20191000TkId('TTbar_Tauola_14TeV_cfi',Kby(9,100))
steps['TTbar_UPG2019_14_3000']=gen20193000('TTbar_Tauola_14TeV_cfi',Kby(9,100))

steps['TTbar_UPG2019PU20_14_STAR']=gen2019start('TTbar_Tauola_14TeV_cfi',Kby(9,100))
steps['TTbar_UPG2019PU20_14_300']=gen2019300('TTbar_Tauola_14TeV_cfi',Kby(9,100))
steps['TTbar_UPG2019PU20_14_500']=gen2019500('TTbar_Tauola_14TeV_cfi',Kby(9,100))
steps['TTbar_UPG2019PU20_14_1000']=gen20191000('TTbar_Tauola_14TeV_cfi',Kby(9,100))

#2023

step1Up2023_BE_Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'auto:upgradePLS3', 
                             '--beamspot' : 'Gauss',
                             '--magField' : '38T_PostLS1',
                             '--datatier' : 'GEN-SIM',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'ExtendedPhase2TkBE',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1,SLHCUpgradeSimulations/Configuration/phase2TkCustomsBE.customise'
                             }
def gen2023_BE(fragment,howMuch):
    global step1Up2023_BE_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2023_BE_Defaults])
    
steps['FourMuPt1_200_UPG2023_BE']=gen2023_BE('FourMuPt_1_200_cfi',Kby(10,100))
steps['MinBias_TuneZ2star_UPG2023_14_BE']=gen2023_BE('MinBias_TuneZ2star_14TeV_pythia6_cff',Kby(9,300))
steps['TTbar_UPG2023_14_BE']=gen2023_BE('TTbar_Tauola_14TeV_cfi',Kby(9,100))
  
step1Up2023_BE5D_Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'auto:upgradePLS3', 
                             '--beamspot' : 'Gauss',
                             '--datatier' : 'GEN-SIM',
                             '--magField' : '38T_PostLS1',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'ExtendedPhase2TkBE5D',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_phase2_BE5D'
                             }
def gen2023_BE5D(fragment,howMuch):
    global step1Up2023_BE5D_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2023_BE5D_Defaults])
    
steps['FourMuPt1_200_UPG2023_BE5D']=gen2023_BE5D('FourMuPt_1_200_cfi',Kby(10,100))
steps['MinBias_TuneZ2star_UPG2023_14_BE5D']=gen2023_BE5D('MinBias_TuneZ2star_14TeV_pythia6_cff',Kby(9,300))
steps['TTbar_UPG2023_14_BE5D']=gen2023_BE5D('TTbar_Tauola_14TeV_cfi',Kby(9,100))
  
    
step1Up2023_LB4_Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'auto:upgradePLS3', 
                             '--beamspot' : 'Gauss',
                             '--datatier' : 'GEN-SIM',
                             '--magField' : '38T_PostLS1',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'ExtendedPhase2TkLB_4LPS_2L2S',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1,SLHCUpgradeSimulations/Configuration/phase2TkCustoms_LB_4LPS_2L2S.customise'
                             }
def gen2023_LB4(fragment,howMuch):
    global step1Up2023_LB4_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2023_LB4_Defaults])
    
steps['FourMuPt1_200_UPG2023_LB4']=gen2023_LB4('FourMuPt_1_200_cfi',Kby(10,100))
steps['MinBias_TuneZ2star_UPG2023_14_LB4']=gen2023_LB4('MinBias_TuneZ2star_14TeV_pythia6_cff',Kby(9,300))
steps['TTbar_UPG2023_14_LB4']=gen2023_LB4('TTbar_Tauola_14TeV_cfi',Kby(9,100))
  
 
step1Up2023_LB6_Defaults = {'-s' : 'GEN,SIM',
                             '-n' : 10,
                             '--conditions' : 'auto:upgradePLS3', 
                             '--beamspot' : 'Gauss',
                             '--datatier' : 'GEN-SIM',
                             '--magField' : '38T_PostLS1',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry' : 'ExtendedPhase2TkLB_6PS',
                             '--customise' : 'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1,SLHCUpgradeSimulations/Configuration/phase2TkCustoms_LB_6PS.customise'
                             }
def gen2023_LB6(fragment,howMuch):
    global step1Up2023_LB6_Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2023_LB6_Defaults])
    
steps['FourMuPt1_200_UPG2023_LB6']=gen2023_LB6('FourMuPt_1_200_cfi',Kby(10,100))
steps['MinBias_TuneZ2star_UPG2023_14_LB6']=gen2023_LB6('MinBias_TuneZ2star_14TeV_pythia6_cff',Kby(9,300))
steps['TTbar_UPG2023_14_LB6']=gen2023_LB6('TTbar_Tauola_14TeV_cfi',Kby(9,100))
   
## pPb tests
step1PPbDefaults={'--beamspot':'Realistic8TeVCollisionPPbBoost'}
steps['AMPT_PPb_5020GeV_MinimumBias']=merge([{'-n':10},step1PPbDefaults,genS('AMPT_PPb_5020GeV_MinimumBias_cfi',Kby(9,100))])
steps['AMPT_PPb_5020GeV_MinimumBiasINPUT']={'INPUT':InputInfo(dataSet='/RelValAMPT_PPb_5020GeV_MinimumBias/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}

## heavy ions tests
U500by1={'--relval': '500,1'}
U80by1={'--relval': '80,1'}

hiDefaults={'--conditions':'auto:starthi_HIon',
           '--scenario':'HeavyIons'}

steps['HydjetQ_MinBias_2760GeV']=merge([{'-n':1},hiDefaults,genS('Hydjet_Quenched_MinBias_2760GeV_cfi',U500by1)])
steps['HydjetQ_MinBias_2760GeVINPUT']={'INPUT':InputInfo(dataSet='/RelValHydjetQ_MinBias_2760GeV/%s/GEN-SIM'%(baseDataSetRelease[1],),location='STD',split=5)}
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
step1FastDefaults =merge([{'-s':'GEN,SIM,RECO,EI,HLT:@relval,VALIDATION',
                           '--fast':'',
                           '--eventcontent':'FEVTDEBUGHLT,DQM',
                           '--datatier':'GEN-SIM-DIGI-RECO,DQM',
                           '--relval':'27000,3000'},
                          step1Defaults])

steps['TTbarFS']=merge([{'cfg':'TTbar_Tauola_8TeV_cfi'},Kby(100,1000),step1FastDefaults])
steps['SingleMuPt1FS']=merge([{'cfg':'SingleMuPt1_cfi'},step1FastDefaults])
steps['SingleMuPt10FS']=merge([{'cfg':'SingleMuPt10_cfi'},step1FastDefaults])
steps['SingleMuPt100FS']=merge([{'cfg':'SingleMuPt100_cfi'},step1FastDefaults])
steps['SinglePiPt1FS']=merge([{'cfg':'SinglePiPt1_cfi'},step1FastDefaults])
steps['SinglePiPt10FS']=merge([{'cfg':'SinglePiPt10_cfi'},step1FastDefaults])
steps['SinglePiPt100FS']=merge([{'cfg':'SinglePiPt100_cfi'},step1FastDefaults])
steps['ZEEFS']=merge([{'cfg':'ZEE_8TeV_cfi'},Kby(100,2000),step1FastDefaults])
steps['ZTTFS']=merge([{'cfg':'ZTT_Tauola_OneLepton_OtherHadrons_8TeV_cfi'},Kby(100,2000),step1FastDefaults])
steps['QCDFlatPt153000FS']=merge([{'cfg':'QCDForPF_8TeV_cfi'},Kby(27,2000),step1FastDefaults])
steps['QCD_Pt_80_120FS']=merge([{'cfg':'QCD_Pt_80_120_8TeV_cfi'},Kby(100,500),stCond,step1FastDefaults])
steps['QCD_Pt_3000_3500FS']=merge([{'cfg':'QCD_Pt_3000_3500_8TeV_cfi'},Kby(100,500),stCond,step1FastDefaults])
steps['H130GGgluonfusionFS']=merge([{'cfg':'H130GGgluonfusion_8TeV_cfi'},step1FastDefaults])
steps['SingleGammaFlatPt10To10FS']=merge([{'cfg':'SingleGammaFlatPt10To100_cfi'},Kby(100,500),step1FastDefaults])

steps['TTbarSFS']=merge([{'cfg':'TTbar_Tauola_8TeV_cfi'},
                        {'-s':'GEN,SIM',
                         '--eventcontent':'FEVTDEBUG',
                         '--datatier':'GEN-SIM',
                         '--fast':''},
                        step1Defaults])
steps['TTbarSFSA']=merge([{'cfg':'TTbar_Tauola_8TeV_cfi',
                           '-s':'GEN,SIM,RECO,EI,HLT,VALIDATION',
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
steps['ReggeGribovPartonMC_EposLHC_5TeV_pPb']=genvalid('GeneratorInterface/ReggeGribovPartonMCInterface/ReggeGribovPartonMC_EposLHC_5TeV_pPb_cfi',step1GenDefaults)

PU={'-n':10,'--pileup':'default','--pileup_input':'dbs:/RelValMinBias/%s/GEN-SIM'%(baseDataSetRelease[2],)}
PUFS={'--pileup':'default'}
PUFS2={'--pileup':'mix_2012_Startup_inTimeOnly'}
steps['TTbarFSPU']=merge([PUFS,Kby(100,500),steps['TTbarFS']] )
steps['TTbarFSPU2']=merge([PUFS2,Kby(100,500),steps['TTbarFS']])
##########################

#### fastsim section for phase2 ####
##no forseen to do things in two steps GEN-SIM then FASTIM->end: maybe later
step1FastDefaultsP1 =merge([{'-s':'GEN,SIM,RECO,VALIDATION',
                           '--eventcontent':'FEVTDEBUGHLT,DQM',
                           '--datatier':'GEN-SIM-DIGI-RECO,DQM',
                           '--conditions':'auto:upgradePLS3', 
			   '--fast':'',
			   '--geometry' : 'Extended2017',
			   '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.fastsimDefault',
                           '--relval':'27000,3000'},
                          step1Defaults])

steps['TTbarFSP1']=merge([{'cfg':'TTbar_Tauola_14TeV_cfi'},Kby(100,1000),step1FastDefaultsP1])
steps['TTbar8FSP1']=merge([{'cfg':'TTbar_Tauola_8TeV_cfi'},Kby(100,1000),step1FastDefaultsP1])
steps['SingleMuPt1FSP1']=merge([{'cfg':'SingleMuPt1_cfi'},step1FastDefaultsP1])
steps['SingleMuPt10FSP1']=merge([{'cfg':'SingleMuPt10_cfi'},step1FastDefaultsP1])
steps['SingleMuPt100FSP1']=merge([{'cfg':'SingleMuPt100_cfi'},step1FastDefaultsP1])
steps['MinBias_TuneZ2starFSP1']=merge([{'cfg':'MinBias_TuneZ2star_14TeV_pythia6_cff'},step1FastDefaultsP1])
steps['MinBias_TuneZ2star8FSP1']=merge([{'cfg':'MinBias_TuneZ2star_8TeV_pythia6_cff'},step1FastDefaultsP1])

step1FastDefaultsP1PU =merge([{'-s':'GEN,SIM,RECO,VALIDATION',
                           '--eventcontent':'FEVTDEBUGHLT,DQM',
                           '--datatier':'GEN-SIM-DIGI-RECO,DQM',
                           '--conditions':'auto:upgradePLS3', 
			   '--fast':'',
			   '--pileup':'default',
			   '--geometry' : 'Extended2017',
			   '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.fastsimDefault',
                           '--relval':'27000,3000'},
                          step1Defaults])

steps['TTbar8FSPUP1']=merge([{'cfg':'TTbar_Tauola_8TeV_cfi'},Kby(100,1000),step1FastDefaultsP1PU])

step1FastDefaultsP2 =merge([{'-s':'GEN,SIM,RECO,VALIDATION',
                           '--eventcontent':'FEVTDEBUGHLT,DQM',
                           '--datatier':'GEN-SIM-DIGI-RECO,DQM',
			   '--fast':'',
                           '--conditions':'auto:upgradePLS3', 
			   '--geometry' : 'ExtendedPhase2TkBE',
			   '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.fastsimPhase2',
                           '--relval':'27000,3000'},
                          step1Defaults])

steps['TTbarFSP2']=merge([{'cfg':'TTbar_Tauola_14TeV_cfi'},Kby(100,1000),step1FastDefaultsP2])
steps['TTbar8FSP2']=merge([{'cfg':'TTbar_Tauola_8TeV_cfi'},Kby(100,1000),step1FastDefaultsP2])
steps['SingleMuPt1FSP2']=merge([{'cfg':'SingleMuPt1_cfi'},step1FastDefaultsP2])
steps['SingleMuPt10FSP2']=merge([{'cfg':'SingleMuPt10_cfi'},step1FastDefaultsP2])
steps['SingleMuPt100FSP2']=merge([{'cfg':'SingleMuPt100_cfi'},step1FastDefaultsP2])
steps['MinBias_TuneZ2starFSP2']=merge([{'cfg':'MinBias_TuneZ2star_14TeV_pythia6_cff'},step1FastDefaultsP2])
steps['MinBias_TuneZ2star8FSP2']=merge([{'cfg':'MinBias_TuneZ2star_8TeV_pythia6_cff'},step1FastDefaultsP2])


step1FastDefaultsP2PU =merge([{'-s':'GEN,SIM,RECO,VALIDATION',
                           '--eventcontent':'FEVTDEBUGHLT,DQM',
                           '--datatier':'GEN-SIM-DIGI-RECO,DQM',
			   '--fast':'',
			   '--pileup':'default',
                           '--conditions':'auto:upgradePLS3', 
			   '--geometry' : 'ExtendedPhase2TkBE',
			   '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.fastsimPhase2',
                           '--relval':'27000,3000'},
                          step1Defaults])
steps['TTbar8FSPUP2']=merge([{'cfg':'TTbar_Tauola_8TeV_cfi'},Kby(100,1000),step1FastDefaultsP2PU])

step1FastDefaultsP2Forw =merge([{'-s':'GEN,SIM,RECO,VALIDATION',
                           '--eventcontent':'FEVTDEBUGHLT,DQM',
                           '--datatier':'GEN-SIM-DIGI-RECO,DQM',
                           '--conditions':'auto:upgradePLS3', 
			   '--geometry' : 'ExtendedPhase2TkBEForward',
			   '--fast':'',
			   '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.fastsimDefault',
                           '--relval':'27000,3000'},
                          step1Defaults])

steps['TTbarFSP2Forw']=merge([{'cfg':'TTbar_Tauola_14TeV_cfi'},Kby(100,1000),step1FastDefaultsP2Forw])
steps['SingleMuPt1FSP2Forw']=merge([{'cfg':'SingleMuPt1_cfi'},step1FastDefaultsP2Forw])
steps['SingleMuPt10FSP2Forw']=merge([{'cfg':'SingleMuPt10_cfi'},step1FastDefaultsP2Forw])
steps['SingleMuPt100FSP2Forw']=merge([{'cfg':'SingleMuPt100_cfi'},step1FastDefaultsP2Forw])
steps['MinBias_TuneZ2starFSP2Forw']=merge([{'cfg':'MinBias_TuneZ2star_14TeV_pythia6_cff'},step1FastDefaultsP2Forw])



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

#wmsplit['DIGIHI']=5


#for 2017
step2Upg2017Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'auto:upgrade2017', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
		 '--magField' : '38T_PostLS1',
		 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017',
                 '--geometry' : 'Extended2017'
                  }
steps['DIGIUP17']=merge([step2Upg2017Defaults])

step2Upg2017puDefaults = {'-s':'DIGI,L1,DIGI2RAW',
		 '--conditions':'auto:upgrade2017', 
		 '--datatier':'GEN-SIM-DIGI-RAW',
		 '-n':'10',
		 '--eventcontent':'FEVTDEBUGHLT',
		 '--magField' : '38T_PostLS1',
	       '--pileup': 'AVE_20_BX_25ns',
		 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017',
		 '--geometry' : 'Extended2017',
	       '--pileup_input':'dbs:/RelValMinBias_TuneZ2star_14TeV/CMSSW_6_1_2_SLHC6-DES17_61_V5_UPG2017-v1/GEN-SIM'
		  }
steps['DIGIPUUP17']=merge([step2Upg2017puDefaults])

step2Upg2017EcalFinepuDefaults = {'-s':'DIGI,L1,DIGI2RAW',
		 '--conditions':'auto:upgrade2017', 
		 '--datatier':'GEN-SIM-DIGI-RAW',
		 '-n':'10',
		 '--eventcontent':'FEVTDEBUGHLT',
		 '--magField' : '38T_PostLS1',
	       '--pileup': 'AVE_20_BX_25ns',
		 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/pileUp_MinBias_TuneZ2star_14TeV_pythia6_620SLHC4_UPG2017ECALFINE.customisePU',
		 '--geometry' : 'Extended2017'
#	       '--pileup_input':'dbs:/RelValMinBias_TuneZ2star_14TeV/CMSSW_6_1_2_SLHC6-DES17_61_V5_UPG2017-v1/GEN-SIM'
		  }
steps['DIGIPUUP17ECALFINE']=merge([step2Upg2017EcalFinepuDefaults])

#add this line when testing from an input file that is not strictly GEN-SIM
#addForAll(step2,{'--process':'DIGI'})

#for 2019
step2Upg2019Defaults = {'-s':'DIGI,L1,DIGI2RAW',
		 '--conditions':'auto:upgrade2019', 
		 '--datatier':'GEN-SIM-DIGI-RAW',
		 '-n':'10',
		 '--eventcontent':'FEVTDEBUGHLT',
		 '--magField' : '38T_PostLS1',
		 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019',
		 '--geometry' : 'Extended2019'
		  }
steps['DIGIUP19']=merge([step2Upg2019Defaults])


step2Upg2019puDefaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'auto:upgrade2019', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--magField' : '38T_PostLS1',
		 '--pileup': 'AVE_20_BX_25ns',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019',
                 '--geometry' : 'Extended2017',
		 '--pileup_input':'dbs:/RelValMinBias_TuneZ2star_14TeV/CMSSW_6_1_2_SLHC6-DES19_61_V5_UPG2019-v1/GEN-SIM'
                  }
steps['DIGIPUUP19']=merge([step2Upg2019puDefaults])



####DIGI AGING VALIDATION - DESIGN set of reference


####DIGI AGING VALIDATION - STARTUP set of reference

step2Upg2017_START_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W17_150_62E2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--magField' : '38T_PostLS1',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017',
                 '--geometry' : 'Extended2017'
                  }
steps['DIGIUP17STAR']=merge([step2Upg2017_START_Defaults])



step2Upg2017pu_START_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W17_150_62E2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT',
		 '--pileup': 'AVE_20_BX_25ns',
                 '--magField' : '38T_PostLS1',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017',
                 '--geometry' : 'Extended2017',
		 '--pileup_input':'dbs:/RelValMinBias_TuneZ2star_14TeV/CMSSW_6_1_2_SLHC6-DES17_61_V5_UPG2017-v1/GEN-SIM'
                  }
steps['DIGIPUUP17STAR']=merge([step2Upg2017pu_START_Defaults])



####DIGI AGING VALIDATION - 300fb-1

step2Upg2017_300_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W17_300_62E2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_300',
                 '--geometry' : 'Extended2017'
                  }
steps['DIGIUP17300']=merge([step2Upg2017_300_Defaults])

step2Upg2017pu_300_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W17_300_62E2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--magField' : '38T_PostLS1',
		 '--pileup': 'AVE_20_BX_25ns',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_300',
                 '--geometry' : 'Extended2017',
		 '--pileup_input':'dbs:/RelValMinBias_TuneZ2star_14TeV/CMSSW_6_1_2_SLHC6-DES17_61_V5_UPG2017-v1/GEN-SIM'
                  }
steps['DIGIPUUP17300']=merge([step2Upg2017pu_300_Defaults])


####DIGI AGING VALIDATION - 300fb-1  COMPLETE ECAK

step2Upg2017_300comp_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W17_300_62C2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_300,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_300',
                 '--geometry' : 'Extended2017'
                  }
steps['DIGIUP17300COMP']=merge([step2Upg2017_300comp_Defaults])

####DIGI AGING VALIDATION - 500fb-1 

step2Upg2017_500_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W17_500_62E2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_500',
                 '--geometry' : 'Extended2017'
                  }
steps['DIGIUP17500']=merge([step2Upg2017_500_Defaults])


step2Upg2017pu_500_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W17_500_62E2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--magField' : '38T_PostLS1',
		 '--pileup': 'AVE_20_BX_25ns',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_500',
                 '--geometry' : 'Extended2017',
		 '--pileup_input':'dbs:/RelValMinBias_TuneZ2star_14TeV/CMSSW_6_1_2_SLHC6-DES17_61_V5_UPG2017-v1/GEN-SIM'
                  }
steps['DIGIPUUP17500']=merge([step2Upg2017pu_500_Defaults])



####DIGI AGING VALIDATION - 1000fb-1 

step2Upg2017_1000_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W17_100062E2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000',
                 '--geometry' : 'Extended2017'
                  }
steps['DIGIUP171000']=merge([step2Upg2017_1000_Defaults])

step2Upg2017pu_1000_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W17_100062E2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT',
		 '--pileup': 'AVE_20_BX_25ns',
                 '--magField' : '38T_PostLS1',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000',
                 '--geometry' : 'Extended2017',
		 '--pileup_input':'dbs:/RelValMinBias_TuneZ2star_14TeV/CMSSW_6_1_2_SLHC6-DES17_61_V5_UPG2017-v1/GEN-SIM'
                  }
steps['DIGIPUUP171000']=merge([step2Upg2017pu_1000_Defaults])

####DIGI AGING VALIDATION - 1000fb-1 COMPLETE ECAL

step2Upg2017_1000comp_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W17_100062C2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_1000',
                 '--geometry' : 'Extended2017'
                  }
steps['DIGIUP171000COMP']=merge([step2Upg2017_1000comp_Defaults])

####DIGI AGING VALIDATION - 1000fb-1 tkid

step2Upg2017_1000_TkId_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W17_100062E2A::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000',
                 '--geometry' : 'Extended2017'
                  }
steps['DIGIUP171000TkId']=merge([step2Upg2017_1000_TkId_Defaults])

####DIGI AGING VALIDATION - 1000fb-1 tkid COMPLETE ECAL

step2Upg2017_1000comp_TkId_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W17_100062C2A::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--magField' : '38T_PostLS1',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_1000',
                 '--geometry' : 'Extended2017'
                  }
steps['DIGIUP171000COMPTkId']=merge([step2Upg2017_1000comp_TkId_Defaults])


####DIGI AGING VALIDATION - 3000fb-1 _ 

step2Upg2017_3000_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W17_300062E2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_3000',
                 '--geometry' : 'Extended2017'
                  }
steps['DIGIUP173000']=merge([step2Upg2017_3000_Defaults])


####DIGI AGING VALIDATION - 3000fb-1 _ COMPLETE ECAL

step2Upg2017_3000comp_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W17_300062C2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_3000,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_3000',
                 '--geometry' : 'Extended2017'
                  }
steps['DIGIUP173000COMP']=merge([step2Upg2017_3000comp_Defaults])

####DIGI AGING VALIDATION - STARTUP set of reference

step2Upg2019_START_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W19_150_62E2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019',
                 '--geometry' : 'Extended2019'
                  }
steps['DIGIUP19STAR']=merge([step2Upg2019_START_Defaults])



step2Upg2019pu_START_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W19_150_62E2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT',
		 '--pileup': 'AVE_20_BX_25ns',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019',
                 '--geometry' : 'Extended2019',
		 '--pileup_input':'dbs:/RelValMinBias_TuneZ2star_14TeV/CMSSW_6_1_2_SLHC6-DES19_61_V5_UPG2019-v1/GEN-SIM'
                  }
steps['DIGIPUUP19STAR']=merge([step2Upg2019pu_START_Defaults])



####DIGI AGING VALIDATION - 300fb-1

step2Upg2019_300_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W19_300_62E2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_300',
                 '--geometry' : 'Extended2019'
                  }
steps['DIGIUP19300']=merge([step2Upg2019_300_Defaults])

step2Upg2019pu_300_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W19_300_62E2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT',
		 '--pileup': 'AVE_20_BX_25ns',
                 '--magField' : '38T_PostLS1',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_300',
                 '--geometry' : 'Extended2019',
		 '--pileup_input':'dbs:/RelValMinBias_TuneZ2star_14TeV/CMSSW_6_1_2_SLHC6-DES19_61_V5_UPG2019-v1/GEN-SIM'
                  }
steps['DIGIPUUP19300']=merge([step2Upg2019pu_300_Defaults])


####DIGI AGING VALIDATION - 300fb-1  COMPLETE ECAK

step2Upg2019_300comp_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W19_300_62C2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_300,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_300',
                 '--geometry' : 'Extended2019'
                  }
steps['DIGIUP19300COMP']=merge([step2Upg2019_300comp_Defaults])

####DIGI AGING VALIDATION - 500fb-1 

step2Upg2019_500_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W19_500_62E2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--magField' : '38T_PostLS1',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_500',
                 '--geometry' : 'Extended2019'
                  }
steps['DIGIUP19500']=merge([step2Upg2019_500_Defaults])


step2Upg2019pu_500_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W19_500_62E2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--magField' : '38T_PostLS1',
		 '--pileup': 'AVE_20_BX_25ns',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_500',
                 '--geometry' : 'Extended2019',
		 '--pileup_input':'dbs:/RelValMinBias_TuneZ2star_14TeV/CMSSW_6_1_2_SLHC6-DES19_61_V5_UPG2019-v1/GEN-SIM'
                  }
steps['DIGIPUUP19500']=merge([step2Upg2019pu_500_Defaults])



####DIGI AGING VALIDATION - 1000fb-1 

step2Upg2019_1000_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W19_100062E2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000',
                 '--geometry' : 'Extended2019'
                  }
steps['DIGIUP191000']=merge([step2Upg2019_1000_Defaults])

step2Upg2019pu_1000_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W19_100062E2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT',
		 '--pileup': 'AVE_20_BX_25ns',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000',
                 '--geometry' : 'Extended2019',
		 '--pileup_input':'dbs:/RelValMinBias_TuneZ2star_14TeV/CMSSW_6_1_2_SLHC6-DES19_61_V5_UPG2019-v1/GEN-SIM'
                  }
steps['DIGIPUUP191000']=merge([step2Upg2019pu_1000_Defaults])

####DIGI AGING VALIDATION - 1000fb-1 COMPLETE ECAL

step2Upg2019_1000comp_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W19_100062C2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_1000',
                 '--geometry' : 'Extended2019'
                  }
steps['DIGIUP191000COMP']=merge([step2Upg2019_1000comp_Defaults])

####DIGI AGING VALIDATION - 1000fb-1 tkid

step2Upg2019_1000_TkId_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W19_100062E2A::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000',
                 '--geometry' : 'Extended2019'
                  }
steps['DIGIUP191000TkId']=merge([step2Upg2019_1000_TkId_Defaults])

####DIGI AGING VALIDATION - 1000fb-1 tkid COMPLETE ECAL

step2Upg2019_1000comp_TkId_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W19_100062C2A::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_1000',
                 '--geometry' : 'Extended2019'
                  }
steps['DIGIUP191000COMPTkId']=merge([step2Upg2019_1000comp_TkId_Defaults])


####DIGI AGING VALIDATION - 3000fb-1 _ 

step2Upg2019_3000_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W19_300062E2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_3000',
                 '--geometry' : 'Extended2019'
                  }
steps['DIGIUP193000']=merge([step2Upg2019_3000_Defaults])


####DIGI AGING VALIDATION - 3000fb-1 _ COMPLETE ECAL

step2Upg2019_3000comp_Defaults = {'-s':'DIGI,L1,DIGI2RAW',
                 '--conditions':'W19_300062C2::All', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_3000,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_3000',
                 '--geometry' : 'Extended2019'
                  }
steps['DIGIUP193000COMP']=merge([step2Upg2019_3000comp_Defaults])



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


dataReco={'--conditions':'auto:com10',
          '-s':'RAW2DIGI,L1Reco,RECO,EI,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias,DQM',
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
#wmsplit['HLTD']=5

steps['RECOD']=merge([{'--scenario':'pp',},dataReco])
steps['RECOSKIMALCA']=merge([{'--inputCommands':'"keep *","drop *_*_*_RECO"'
                              },steps['RECOD']])
steps['RECOSKIM']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,EI,DQM',
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
                       '-s':'RAW2DIGI,L1Reco,RECO,EI,ALCAPRODUCER:@allForPrompt,DQM,ENDJOB',
                       '--datatier':'RECO,AOD,ALCARECO,DQMROOT',
                       '--eventcontent':'RECO,AOD,ALCARECO,DQMROOT',
                       '--process':'RECO'
                       },dataReco])
steps['TIER0EXP']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,EI,ALCAPRODUCER:@allForExpress,DQM,ENDJOB',
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
                  '-s'            : 'RAW2DIGI,L1Reco,RECO,EI,VALIDATION,DQM',
                  '--conditions'  : 'auto:startup',
                  '--no_exec'     : '',
                  '--datatier'    : 'GEN-SIM-RECO,DQM',
                  '--eventcontent': 'RECOSIM,DQM'
                  }

steps['DIGIPU']=merge([{'--process':'REDIGI'},steps['DIGIPU1']])
#wmsplit['DIGIPU']=4
#wmsplit['DIGIPU1']=4

steps['RECODreHLT']=merge([{'--hltProcess':'reHLT','--conditions':'auto:com10_%s'%menu},steps['RECOD']])
#wmsplit['RECODreHLT']=2

steps['RECO']=merge([step3Defaults])
steps['RECODBG']=merge([{'--eventcontent':'RECODEBUG,DQM'},steps['RECO']])
steps['RECOPROD1']=merge([{ '-s' : 'RAW2DIGI,L1Reco,RECO,EI', '--datatier' : 'GEN-SIM-RECO,AODSIM', '--eventcontent' : 'RECOSIM,AODSIM'},step3Defaults])
steps['RECOCOS']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCA:MuAlCalIsolatedMu,DQM','--scenario':'cosmics'},stCond,step3Defaults])
steps['RECOMIN']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,EI,ALCA:SiStripCalZeroBias+SiStripCalMinBias+EcalCalPhiSym+EcalCalPi0Calib+EcalCalEtaCalib,VALIDATION,DQM'},stCond,step3Defaults])

steps['RECODDQM']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,EI,DQM:@common+@muon+@hcal+@jetmet+@ecal'},steps['RECOD']])

steps['RECOPU1']=merge([PU,steps['RECO']])
#wmsplit['RECOPU1']=1
steps['RECOPUDBG']=merge([{'--eventcontent':'RECODEBUG,DQM'},steps['RECOPU1']])
steps['RERECOPU1']=merge([{'--hltProcess':'REDIGI'},steps['RECOPU1']])

steps['RECO_ID']=merge([{'--hltProcess':'HLT2'},steps['RECO']])

steps['RECOHI']=merge([hiDefaults,{'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM'},step3Defaults])
#wmsplit['RECOHI']=5

steps['DIGIHISt3']=steps['DIGIHI']

steps['RECOHID11St3']=merge([{
                              '--process':'ZStoRECO'},
                             steps['RECOHID11']])
steps['RECOHIR10D11']=merge([{'--filein':'file:step2_inREPACKRAW.root',
                              '--filtername':'reRECO'},
                             steps['RECOHID11St3']])
steps['RECOFS']=merge([{'--fast':'',
                        '-s':'RECO,EI,HLT:@relval,VALIDATION'},
                       steps['RECO']])

#for 2017
step3Up2017Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,EI,VALIDATION,DQM',
                 '--conditions':'auto:upgrade2017', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                  '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017',
                 '--geometry' : 'Extended2017'
                 }
                             
steps['RECOUP17']=merge([step3Up2017Defaults])

step3Up2017PUEcalFineDefaults = {'-s':'RAW2DIGI,L1Reco,RECO',
                                 '--conditions':'auto:upgrade2017', 
                                 '--datatier':'GEN-SIM-RECO',
                                 '-n':'10',
                                 '--magField' : '38T_PostLS1',
                                 '--eventcontent':'FEVTDEBUGHLT',
                                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017',
                                 '--geometry' : 'Extended2017',
                                 '--pileup': 'AVE_20_BX_25ns',
                                 '--customise': 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/pileUp_MinBias_TuneZ2star_14TeV_pythia6_620SLHC4_UPG2017ECALFINE.customisePU'
                 }
                             
steps['RECOPUUP17ECALFINE']=merge([step3Up2017PUEcalFineDefaults])




#for 2019
step3Up2019Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'auto:upgrade2019', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019',
                 '--geometry' : 'Extended2019'
                 }
                             
steps['RECOUP19']=merge([step3Up2019Defaults])




####RECO AGING VALIDATION - STARTUP set of reference



step3Up2017_START_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W17_150_62E2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017',
                 '--geometry' : 'Extended2017'
                 }
                             
steps['RECOUP17STAR']=merge([step3Up2017_START_Defaults])

step3Up2017pu_START_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W17_150_62E2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
		 '--pileup': 'AVE_20_BX_25ns',
                 '--magField' : '38T_PostLS1',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017',
                 '--geometry' : 'Extended2017',
		 '--pileup_input':'dbs:/RelValMinBias_TuneZ2star_14TeV/CMSSW_6_1_2_SLHC6-DES17_61_V5_UPG2017-v1/GEN-SIM'
                 }
                             
steps['RECOPUUP17STAR']=merge([step3Up2017pu_START_Defaults])


####RECO AGING VALIDATION - 300fb-1 



step3Up2017_300_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W17_300_62E2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_300',
                 '--geometry' : 'Extended2017'
                 }
                             
steps['RECOUP17300']=merge([step3Up2017_300_Defaults])

step3Up2017pu_300_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W17_300_62E2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
		 '--pileup': 'AVE_20_BX_25ns',
                 '--magField' : '38T_PostLS1',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_300',
                 '--geometry' : 'Extended2017',
		 '--pileup_input':'dbs:/RelValMinBias_TuneZ2star_14TeV/CMSSW_6_1_2_SLHC6-DES17_61_V5_UPG2017-v1/GEN-SIM'
                 }
                             
steps['RECOPUUP17300']=merge([step3Up2017pu_300_Defaults])

####RECO AGING VALIDATION - 300fb-1 COMPLETE ECAL



step3Up2017_300comp_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W17_300_62C2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--magField' : '38T_PostLS1',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_300,SLHCUpgradeSimulations/Configuration/aging.customise_aging_300',
                 '--geometry' : 'Extended2017'
                 }
                             
steps['RECOUP17300COMP']=merge([step3Up2017_300comp_Defaults])


####RECO AGING VALIDATION - 500fb-1 


step3Up2017_500_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W17_500_62E2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_500',
                 '--geometry' : 'Extended2017'
                 }
                             
steps['RECOUP17500']=merge([step3Up2017_500_Defaults])

step3Up2017pu_500_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W17_500_62E2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--magField' : '38T_PostLS1',
		 '--pileup': 'AVE_20_BX_25ns',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_500',
                 '--geometry' : 'Extended2017',
		 '--pileup_input':'dbs:/RelValMinBias_TuneZ2star_14TeV/CMSSW_6_1_2_SLHC6-DES17_61_V5_UPG2017-v1/GEN-SIM'
                 }
                             
steps['RECOPUUP17500']=merge([step3Up2017pu_500_Defaults])




####RECO AGING VALIDATION - 1000fb-1 


step3Up2017_1000_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W17_100062E2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--magField' : '38T_PostLS1',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000',
                 '--geometry' : 'Extended2017'
                 }
                             
steps['RECOUP171000']=merge([step3Up2017_1000_Defaults])

step3Up2017pu_1000_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W17_100062E2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
		 '--pileup': 'AVE_20_BX_25ns',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000',
                 '--geometry' : 'Extended2017',
		 '--pileup_input':'dbs:/RelValMinBias_TuneZ2star_14TeV/CMSSW_6_1_2_SLHC6-DES17_61_V5_UPG2017-v1/GEN-SIM'
                 }
                             
steps['RECOPUUP171000']=merge([step3Up2017pu_1000_Defaults])
####RECO AGING VALIDATION - 1000fb-1 COMPLET ECAL


step3Up2017_1000comp_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W17_100062C2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '--magField' : '38T_PostLS1',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_1000,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000',
                 '--geometry' : 'Extended2017'
                 }
                             
steps['RECOUP171000COMP']=merge([step3Up2017_1000comp_Defaults])

####RECO AGING VALIDATION - 1000fb-1 tkId 

step3Up2017_1000_TkId_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W17_100062E2A::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000',
                 '--geometry' : 'Extended2017'
                 }
                             
steps['RECOUP171000TkId']=merge([step3Up2017_1000_TkId_Defaults])

####RECO AGING VALIDATION - 1000fb-1 tkId COMPLETE ECAL

step3Up2017_1000comp_TkId_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W17_100062C2A::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--magField' : '38T_PostLS1',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_1000,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000',
                 '--geometry' : 'Extended2017'
                 }
                             
steps['RECOUP171000COMPTkId']=merge([step3Up2017_1000comp_TkId_Defaults])

####RECO AGING VALIDATION - 3000fb-1 


step3Up2017_3000_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W17_300062E2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--magField' : '38T_PostLS1',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_3000',
                 '--geometry' : 'Extended2017'
                 }
                             
steps['RECOUP173000']=merge([step3Up2017_3000_Defaults])

####RECO AGING VALIDATION - 3000fb-1 COMPLET ECAL


step3Up2017_3000comp_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W17_300062C2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--magField' : '38T_PostLS1',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_3000,SLHCUpgradeSimulations/Configuration/aging.customise_aging_3000',
                 '--geometry' : 'Extended2017'
                 }
                             
steps['RECOUP173000COMP']=merge([step3Up2017_3000comp_Defaults])

####RECO AGING VALIDATION - STARTUP set of reference



step3Up2019_START_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W19_150_62E2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019',
                 '--geometry' : 'Extended2019'
                 }
                             
steps['RECOUP19STAR']=merge([step3Up2019_START_Defaults])

step3Up2019pu_START_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W19_150_62E2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
		 '--pileup': 'AVE_20_BX_25ns',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019',
                 '--magField' : '38T_PostLS1',
                 '--geometry' : 'Extended2019',
		 '--pileup_input':'dbs:/RelValMinBias_TuneZ2star_14TeV/CMSSW_6_1_2_SLHC6-DES19_61_V5_UPG2019-v1/GEN-SIM'
                 }
                             
steps['RECOPUUP19STAR']=merge([step3Up2019pu_START_Defaults])


####RECO AGING VALIDATION - 300fb-1 



step3Up2019_300_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W19_300_62E2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--magField' : '38T_PostLS1',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_300',
                 '--geometry' : 'Extended2019'
                 }
                             
steps['RECOUP19300']=merge([step3Up2019_300_Defaults])

step3Up2019pu_300_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W19_300_62E2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--magField' : '38T_PostLS1',
		 '--pileup': 'AVE_20_BX_25ns',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_300',
                 '--geometry' : 'Extended2019',
		 '--pileup_input':'dbs:/RelValMinBias_TuneZ2star_14TeV/CMSSW_6_1_2_SLHC6-DES19_61_V5_UPG2019-v1/GEN-SIM'
                 }
                             
steps['RECOPUUP19300']=merge([step3Up2019pu_300_Defaults])

####RECO AGING VALIDATION - 300fb-1 COMPLETE ECAL



step3Up2019_300comp_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W19_300_62C2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '--magField' : '38T_PostLS1',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_300,SLHCUpgradeSimulations/Configuration/aging.customise_aging_300',
                 '--geometry' : 'Extended2019'
                 }
                             
steps['RECOUP19300COMP']=merge([step3Up2019_300comp_Defaults])


####RECO AGING VALIDATION - 500fb-1 


step3Up2019_500_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W19_500_62E2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_500',
                 '--geometry' : 'Extended2019'
                 }
                             
steps['RECOUP19500']=merge([step3Up2019_500_Defaults])

step3Up2019pu_500_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W19_500_62E2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
		 '--pileup': 'AVE_20_BX_25ns',
                 '--magField' : '38T_PostLS1',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_500',
                 '--geometry' : 'Extended2019',
		 '--pileup_input':'dbs:/RelValMinBias_TuneZ2star_14TeV/CMSSW_6_1_2_SLHC6-DES19_61_V5_UPG2019-v1/GEN-SIM'
                 }
                             
steps['RECOPUUP19500']=merge([step3Up2019pu_500_Defaults])




####RECO AGING VALIDATION - 1000fb-1 


step3Up2019_1000_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W19_100062E2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000',
                 '--geometry' : 'Extended2019'
                 }
                             
steps['RECOUP191000']=merge([step3Up2019_1000_Defaults])

step3Up2019pu_1000_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W19_100062E2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
		 '--pileup': 'AVE_20_BX_25ns',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000',
                 '--geometry' : 'Extended2019',
		 '--pileup_input':'dbs:/RelValMinBias_TuneZ2star_14TeV/CMSSW_6_1_2_SLHC6-DES19_61_V5_UPG2019-v1/GEN-SIM'
                 }
                             
steps['RECOPUUP191000']=merge([step3Up2019pu_1000_Defaults])
####RECO AGING VALIDATION - 1000fb-1 COMPLET ECAL


step3Up2019_1000comp_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W19_100062C2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_1000,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000',
                 '--geometry' : 'Extended2019'
                 }
                             
steps['RECOUP191000COMP']=merge([step3Up2019_1000comp_Defaults])

####RECO AGING VALIDATION - 1000fb-1 tkId 

step3Up2019_1000_TkId_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W19_100062E2A::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000',
                 '--geometry' : 'Extended2019'
                 }
                             
steps['RECOUP191000TkId']=merge([step3Up2019_1000_TkId_Defaults])

####RECO AGING VALIDATION - 1000fb-1 tkId COMPLETE ECAL

step3Up2019_1000comp_TkId_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W19_100062C2A::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--magField' : '38T_PostLS1',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_1000,SLHCUpgradeSimulations/Configuration/aging.customise_aging_1000',
                 '--geometry' : 'Extended2019'
                 }
                             
steps['RECOUP191000COMPTkId']=merge([step3Up2019_1000comp_TkId_Defaults])

####RECO AGING VALIDATION - 3000fb-1 


step3Up2019_3000_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W19_300062E2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_3000',
                 '--geometry' : 'Extended2019'
                 }
                             
steps['RECOUP193000']=merge([step3Up2019_3000_Defaults])

####RECO AGING VALIDATION - 3000fb-1 COMPLET ECAL


step3Up2019_3000comp_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                 '--conditions':'W19_300062C2::All', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/combinedCustoms.ecal_complete_aging_3000,SLHCUpgradeSimulations/Configuration/aging.customise_aging_3000',
                 '--geometry' : 'Extended2019'
                 }
                             
steps['RECOUP193000COMP']=merge([step3Up2019_3000comp_Defaults])

#for 2023 BE
step3Up2023_BE_Defaults = {'-s':'DIGI,L1,DIGI2RAW,L1TrackTrigger,RECO:pixeltrackerlocalreco',
                 '--conditions':'auto:upgradePLS3', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUG',
                 '--magField' : '38T_PostLS1',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1,SLHCUpgradeSimulations/Configuration/phase2TkCustomsBE.customise,SLHCUpgradeSimulations/Configuration/phase2TkCustomsBE.l1EventContent',
                 '--geometry' : 'ExtendedPhase2TkBE'
                 }
                             
steps['RECOUP23_BE']=merge([step3Up2023_BE_Defaults])


#for 2023 BE5D
step3Up2023_BE5D_Defaults = {'-s':'DIGI,L1,DIGI2RAW,L1TrackTrigger,RECO:pixeltrackerlocalreco',
                 '--conditions':'auto:upgradePLS3', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUG',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_phase2_BE5D',
                 '--geometry' : 'ExtendedPhase2TkBE5D'
                 }
                             
steps['RECOUP23_BE5D']=merge([step3Up2023_BE5D_Defaults])


step3Up2023_LB4_Defaults = {'-s':'DIGI,L1,DIGI2RAW,L1TrackTrigger,RECO:pixeltrackerlocalreco',
                 '--conditions':'auto:upgradePLS3', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--eventcontent':'FEVTDEBUG',
                 '--magField' : '38T_PostLS1',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1,SLHCUpgradeSimulations/Configuration/phase2TkCustoms_LB_4LPS_2L2S.customise,SLHCUpgradeSimulations/Configuration/phase2TkCustoms_LB_4LPS_2L2S.l1EventContent',
		 '--geometry' : 'ExtendedPhase2TkLB_4LPS_2L2S'
                 }
                             
steps['RECOUP23_LB4']=merge([step3Up2023_LB4_Defaults])

step3Up2023_LB6_Defaults = {'-s':'DIGI,L1,DIGI2RAW,L1TrackTrigger,RECO:pixeltrackerlocalreco',
                 '--conditions':'auto:upgradePLS3', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUG',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1,SLHCUpgradeSimulations/Configuration/phase2TkCustoms_LB_6PS.customise,SLHCUpgradeSimulations/Configuration/phase2TkCustoms_LB_6PS.l1EventContent',
		 '--geometry' : 'ExtendedPhase2TkLB_6PS'
                 }
                             
steps['RECOUP23_LB6']=merge([step3Up2023_LB6_Defaults])



########################### split be5d into three steps

step2Up2023_BE5D_Defaults = {'-s':'DIGI,L1,L1TrackTrigger,DIGI2RAW',
                 '--conditions':'auto:upgradePLS3', 
                 '--datatier':'GEN-SIM-DIGI-RAW',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_phase2_BE5D',
                 '--geometry' : 'ExtendedPhase2TkBE5D'
                 }
                             
steps['DIGIUP23_BE5D']=merge([step2Up2023_BE5D_Defaults])

step3NUp2023_BE5D_Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,EI,VALIDATION,DQM',
                 '--conditions':'auto:upgradePLS3', 
                 '--datatier':'GEN-SIM-RECO,DQM',
                 '-n':'10',
                 '--magField' : '38T_PostLS1',
                 '--eventcontent':'FEVTDEBUGHLT,DQM',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_phase2_BE5D',
                 '--geometry' : 'ExtendedPhase2TkBE5D'
                 }
                             
steps['NRECOUP23_BE5D']=merge([step3NUp2023_BE5D_Defaults])


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

steps['HARVESTUP17']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                   '--conditions':'auto:upgrade2017', 
                   '--magField' : '38T_PostLS1',
                   '--mc':'',
                   '--customise' : 'SLHCUpgradeSimulations/Configuration/phase1TkCustoms.customise',
		   '--geometry' : 'Extended2017'
                   }
steps['HARVESTUP19']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                   '--conditions':'auto:upgrade2019', 
                   '--mc':'',
                   '--magField' : '38T_PostLS1',
                   '--customise' : 'SLHCUpgradeSimulations/Configuration/phase1TkCustoms.customise',
		   '--geometry' : 'Extended2019'
                   }

steps['HARVESTUPBE5D']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                   '--conditions':'auto:upgradePLS3', 
                   '--mc':'',
                   '--magField' : '38T_PostLS1',
                   '--customise' : 'SLHCUpgradeSimulations/Configuration/phase1TkCustoms.customise',
		   '--geometry' : 'ExtendedPhase2TkBE5D'
                   }

####HARVEST AGING VALIDATION - DESIGN set of reference
steps['HARVESTUP17DES']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                   '--conditions':'DES17_61_V5::All', 
                   '--mc':'',
                   '--magField' : '38T_PostLS1',
                   '--customise' : 'SLHCUpgradeSimulations/Configuration/phase1TkCustoms.customise',
		   '--geometry' : 'Extended2017'
                   }
		   
####HARVEST AGING VALIDATION - STARTUP set of reference
steps['HARVESTUP17STAR']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                   '--conditions':'W17_150_62E2::All', 
                   '--mc':'',
                   '--magField' : '38T_PostLS1',
                   '--customise' : 'SLHCUpgradeSimulations/Configuration/phase1TkCustoms.customise',
		   '--geometry' : 'Extended2017'
                   }
####HARVEST AGING VALIDATION - 300fb-1 


steps['HARVESTUP17300']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                   '--conditions':'W17_300_62E2::All', 
                   '--mc':'',
                   '--magField' : '38T_PostLS1',
                   '--customise' : 'SLHCUpgradeSimulations/Configuration/phase1TkCustoms.customise',
		   '--geometry' : 'Extended2017'
                   }
		   
		   
####HARVEST AGING VALIDATION - 500fb-1 


steps['HARVESTUP17500']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                   '--conditions':'W17_500_62E2::All', 
                   '--mc':'',
                   '--magField' : '38T_PostLS1',
                   '--customise' : 'SLHCUpgradeSimulations/Configuration/phase1TkCustoms.customise',
		   '--geometry' : 'Extended2017'
                   }
		   
		   
####HARVEST AGING VALIDATION - 1000fb-1 


steps['HARVESTUP171000']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                   '--conditions':'W17_100062E2::All', 
                   '--mc':'',
                   '--magField' : '38T_PostLS1',
                   '--customise' : 'SLHCUpgradeSimulations/Configuration/phase1TkCustoms.customise',
		   '--geometry' : 'Extended2017'
                   }

steps['HARVESTUP171000TkId']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                   '--conditions':'W17_100062E2A::All', 
                   '--mc':'',
                   '--magField' : '38T_PostLS1',
                   '--customise' : 'SLHCUpgradeSimulations/Configuration/phase1TkCustoms.customise',
		   '--geometry' : 'Extended2017'
                   }
		   
####HARVEST AGING VALIDATION - 3000fb-1 


steps['HARVESTUP173000']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                   '--conditions':'W19_300062E2::All', 
                   '--mc':'',
                   '--magField' : '38T_PostLS1',
                   '--customise' : 'SLHCUpgradeSimulations/Configuration/phase1TkCustoms.customise',
		   '--geometry' : 'Extended2017'
                   }
		   
		   
####HARVEST AGING VALIDATION - DESIGN set of reference
steps['HARVESTUP19DES']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                   '--conditions':'DES17_61_V5::All', 
                   '--mc':'',
                   '--magField' : '38T_PostLS1',
                   '--customise' : 'SLHCUpgradeSimulations/Configuration/phase1TkCustoms.customise',
		   '--geometry' : 'Extended2019'
                   }
		   
####HARVEST AGING VALIDATION - STARTUP set of reference
steps['HARVESTUP19STAR']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                   '--conditions':'W19_150_62E2::All', 
                   '--mc':'',
                   '--magField' : '38T_PostLS1',
                   '--customise' : 'SLHCUpgradeSimulations/Configuration/phase1TkCustoms.customise',
		   '--geometry' : 'Extended2019'
                   }
####HARVEST AGING VALIDATION - 300fb-1 


steps['HARVESTUP19300']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                   '--conditions':'W19_300_62E2::All', 
                   '--mc':'',
                   '--magField' : '38T_PostLS1',
                   '--customise' : 'SLHCUpgradeSimulations/Configuration/phase1TkCustoms.customise',
		   '--geometry' : 'Extended2019'
                   }
		   
		   
####HARVEST AGING VALIDATION - 500fb-1 


steps['HARVESTUP19500']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                   '--conditions':'W19_500_62E2::All', 
                   '--mc':'',
                   '--magField' : '38T_PostLS1',
                   '--customise' : 'SLHCUpgradeSimulations/Configuration/phase1TkCustoms.customise',
		   '--geometry' : 'Extended2019'
                   }
		   
		   
####HARVEST AGING VALIDATION - 1000fb-1 


steps['HARVESTUP191000']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                   '--conditions':'W19_100062E2::All', 
                   '--mc':'',
                   '--customise' : 'SLHCUpgradeSimulations/Configuration/phase1TkCustoms.customise',
                   '--magField' : '38T_PostLS1',
		   '--geometry' : 'Extended2019'
                   }

steps['HARVESTUP191000TkId']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                   '--conditions':'W19_100062E2A::All', 
                   '--mc':'',
                   '--magField' : '38T_PostLS1',
                   '--customise' : 'SLHCUpgradeSimulations/Configuration/phase1TkCustoms.customise',
		   '--geometry' : 'Extended2019'
                   }
		   
####HARVEST AGING VALIDATION - 3000fb-1 


steps['HARVESTUP193000']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                   '--conditions':'W19_300062E2::All', 
                   '--mc':'',
                   '--magField' : '38T_PostLS1',
                   '--customise' : 'SLHCUpgradeSimulations/Configuration/phase1TkCustoms.customise',
		   '--geometry' : 'Extended2019'
                   }
		   
		   
#fastsim upgrade
	   
steps['HARVESTFSP1']={'-s':'HARVESTING:validationHarvestingFS',
                   '--conditions':'auto:upgradePLS3',
                   '--mc':'',
		   '--geometry' : 'Extended2017',
                   '--scenario':'pp'}
		   
steps['HARVESTFSP2']={'-s':'HARVESTING:validationHarvestingFS',
                   '--conditions':'auto:upgradePLS3',
                   '--mc':'',
		   '--geometry' : 'ExtendedPhase2TkBE',
                   '--scenario':'pp'}
steps['HARVESTFSP2Forw']={'-s':'HARVESTING:validationHarvestingFS',
                   '--conditions':'auto:upgradePLS3',
                   '--mc':'',
		   '--geometry' : 'ExtendedPhase2TkBEForward',
                   '--scenario':'pp'}
		   
		   
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
#                                     '-s':'GEN,SIM,DIGI,L1,DIGI2RAW,HLT:@relval,RAW2DIGI,L1Reco,RECO,EI,ALCA:MuAlCalIsolatedMu+DtCalib,VALIDATION,DQM',
#                                     '--datatier':'GEN-SIM-DIGI-RAW-HLTDEBUG-RECO,DQM',
#                                     '--eventcontent':'FEVTDEBUGHLT,DQM'},
#                                    K9by50,stCond,step1Defaults])
#steps['DIGI2RECO']=merge([{'-s':'DIGI,L1,DIGI2RAW,HLT:@relval,RAW2DIGI,L1Reco,RECO,EI,VALIDATION,DQM',
#                           '--filtername':'DIGItoRECO',
#                           '--process':'RECO',
#                           '--eventcontent':'RECOSIM,DQM',
#                           '--datatier':'GEN-SIM-RECO,DQM',
#                           },
#                            stCond,step3Defaults])
steps['RECOFROMRECO']=merge([{'-s':'RECO,EI',
                              '--filtername':'RECOfromRECO',
                              '--process':'reRECO',
                              '--datatier':'AODSIM',
                              '--eventcontent':'AODSIM',
                              },
                             stCond,step3Defaults])


steps['RECOFROMRECOSt2']=steps['RECOFROMRECO']

steps['RECODFROMRAWRECO']=merge([{'-s':'RAW2DIGI:RawToDigi_noTk,L1Reco,RECO:reconstruction_noTracking,EI',
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
