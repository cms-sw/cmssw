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
        
    def das(self):
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
# 2015 step1 gensim
step1Up2015Defaults = {'-s' : 'GEN,SIM',
                             '-n'            : 10,
                             '--conditions'  : 'auto:upgradePLS1', 
                             '--datatier'    : 'GEN-SIM',
                             '--eventcontent': 'FEVTDEBUG',
                             '--geometry'    : 'Extended2015',
                             '--magField'    : '38T_PostLS1',
                             '--customise'   : 'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1'
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

def gen2015(fragment,howMuch):
    global step1Up2015Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2015Defaults])

### Production test: 13 TeV equivalents
steps['ProdMinBias_13']=gen2015('MinBias_13TeV_cfi',Kby(9,100))
steps['ProdTTbar_13']=gen2015('TTbar_Tauola_13TeV_cfi',Kby(9,100))
steps['ProdQCD_Pt_3000_3500_13']=gen2015('QCD_Pt_3000_3500_13TeV_cfi',Kby(9,100))


steps['MinBias']=gen('MinBias_8TeV_cfi',Kby(9,300))
steps['QCD_Pt_3000_3500']=gen('QCD_Pt_3000_3500_8TeV_cfi',Kby(9,25))
steps['QCD_Pt_600_800']=gen('QCD_Pt_600_800_8TeV_cfi',Kby(9,50))
steps['QCD_Pt_80_120']=gen('QCD_Pt_80_120_8TeV_cfi',Kby(9,100))
steps['MinBias_13']=gen2015('MinBias_13TeV_cfi',Kby(9,300))
steps['QCD_Pt_3000_3500_13']=gen2015('QCD_Pt_3000_3500_13TeV_cfi',Kby(9,25))
steps['QCD_Pt_600_800_13']=gen2015('QCD_Pt_600_800_13TeV_cfi',Kby(9,50))
steps['QCD_Pt_80_120_13']=gen2015('QCD_Pt_80_120_13TeV_cfi',Kby(9,100))

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
steps['SingleElectronPt10_UP15']=gen2015('SingleElectronPt10_cfi',Kby(9,3000))
steps['SingleElectronPt35_UP15']=gen2015('SingleElectronPt35_cfi',Kby(9,500))
steps['SingleElectronPt1000_UP15']=gen2015('SingleElectronPt1000_cfi',Kby(9,50))
steps['SingleElectronFlatPt1To100_UP15']=gen2015('SingleElectronFlatPt1To100_cfi',Mby(2,100))
steps['SingleGammaPt10_UP15']=gen2015('SingleGammaPt10_cfi',Kby(9,3000))
steps['SingleGammaPt35_UP15']=gen2015('SingleGammaPt35_cfi',Kby(9,500))
steps['SingleMuPt1_UP15']=gen2015('SingleMuPt1_cfi',Kby(25,1000))
steps['SingleMuPt10_UP15']=gen2015('SingleMuPt10_cfi',Kby(25,500))
steps['SingleMuPt100_UP15']=gen2015('SingleMuPt100_cfi',Kby(9,500))
steps['SingleMuPt1000_UP15']=gen2015('SingleMuPt1000_cfi',Kby(9,500))
steps['TTbar']=gen('TTbar_Tauola_8TeV_cfi',Kby(9,100))
steps['TTbarLepton']=gen('TTbarLepton_Tauola_8TeV_cfi',Kby(9,100))
steps['ZEE']=gen('ZEE_8TeV_cfi',Kby(9,100))
steps['Wjet_Pt_80_120']=gen('Wjet_Pt_80_120_8TeV_cfi',Kby(9,100))
steps['Wjet_Pt_3000_3500']=gen('Wjet_Pt_3000_3500_8TeV_cfi',Kby(9,50))
steps['LM1_sfts']=gen('LM1_sfts_8TeV_cfi',Kby(9,100))
steps['QCD_FlatPt_15_3000']=gen('QCDForPF_8TeV_cfi',Kby(5,100))
steps['QCD_FlatPt_15_3000HS']=gen('QCDForPF_8TeV_cfi',Kby(50,100))
steps['TTbar_13']=gen2015('TTbar_Tauola_13TeV_cfi',Kby(9,100))
steps['TTbarLepton_13']=gen2015('TTbarLepton_Tauola_13TeV_cfi',Kby(9,100))
steps['ZEE_13']=gen2015('ZEE_13TeV_cfi',Kby(9,100))
steps['Wjet_Pt_80_120_13']=gen2015('Wjet_Pt_80_120_13TeV_cfi',Kby(9,100))
steps['Wjet_Pt_3000_3500_13']=gen2015('Wjet_Pt_3000_3500_13TeV_cfi',Kby(9,50))
steps['LM1_sfts_13']=gen2015('LM1_sfts_13TeV_cfi',Kby(9,100))
steps['QCD_FlatPt_15_3000_13']=gen2015('QCDForPF_13TeV_cfi',Kby(9,100))
steps['QCD_FlatPt_15_3000HS_13']=gen2015('QCDForPF_13TeV_cfi',Kby(50,100))

steps['ZpMM_2250_8TeV_Tauola']=gen('ZpMM_2250_8TeV_Tauola_cfi',Kby(9,100))
steps['ZpEE_2250_8TeV_Tauola']=gen('ZpEE_2250_8TeV_Tauola_cfi',Kby(9,100))
steps['ZpTT_1500_8TeV_Tauola']=gen('ZpTT_1500_8TeV_Tauola_cfi',Kby(9,100))
steps['ZpMM_2250_13TeV_Tauola']=gen2015('ZpMM_2250_13TeV_Tauola_cfi',Kby(9,100))
steps['ZpEE_2250_13TeV_Tauola']=gen2015('ZpEE_2250_13TeV_Tauola_cfi',Kby(9,100))
steps['ZpTT_1500_13TeV_Tauola']=gen2015('ZpTT_1500_13TeV_Tauola_cfi',Kby(9,100))

def identitySim(wf):
    return merge([{'--restoreRND':'SIM','--process':'SIM2'},wf])

steps['SingleMuPt10_ID']=identitySim(steps['SingleMuPt10'])
steps['TTbar_ID']=identitySim(steps['TTbar'])

baseDataSetRelease=[
    'CMSSW_6_2_0_pre8-PRE_ST62_V8-v1', 
    'CMSSW_6_2_0_pre8-PRE_SH62_V15-v1',
    'CMSSW_6_2_0_pre8-PRE_ST62_V8_FastSim-v1',
    'CMSSW_6_2_0_pre8-PRE_SH62_V15-v2',
    'CMSSW_6_1_0_pre6-STARTHI61_V6-v1',
    'CMSSW_6_2_0_pre8-PRE_ST62_V8-v3',
    'CMSSW_6_2_0_patch1-POSTLS162_V1_30Aug2013-v2' # 6_2_0_patch1 for 13 TeV samples
    ]

# note: INPUT commands to be added once GEN-SIM w/ 13TeV+PostLS1Geo will be available 
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
steps['SingleMuPt10IdINPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt10/%s/GEN-SIM-DIGI-RAW-HLTDEBUG'%(baseDataSetRelease[0],),location='STD',split=1)}
steps['SingleMuPt10FSIdINPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt10/%s/GEN-SIM-DIGI-RECO'%(baseDataSetRelease[2],),location='STD',split=1)}
steps['SingleMuPt100INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt100/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['SingleMuPt1000INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt1000/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['TTbarINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['TTbarIdINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/%s/GEN-SIM-DIGI-RAW-HLTDEBUG'%(baseDataSetRelease[0],),location='STD',split=1)}
steps['TTbarFSIdINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/%s/GEN-SIM-DIGI-RECO'%(baseDataSetRelease[2],),location='STD',split=1)}
steps['TTbarLeptonINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbarLepton/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['OldTTbarINPUT']={'INPUT':InputInfo(dataSet='/RelValProdTTbar/CMSSW_5_0_0_pre6-START50_V5-v1/GEN-SIM-RECO',location='STD')}
steps['OldGenSimINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/CMSSW_4_4_2-START44_V7-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',location='STD')}
steps['Wjet_Pt_80_120INPUT']={'INPUT':InputInfo(dataSet='/RelValWjet_Pt_80_120/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['Wjet_Pt_3000_3500INPUT']={'INPUT':InputInfo(dataSet='/RelValWjet_Pt_3000_3500/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['LM1_sftsINPUT']={'INPUT':InputInfo(dataSet='/RelValLM1_sfts/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['QCD_FlatPt_15_3000INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_FlatPt_15_3000/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}

steps['QCD_FlatPt_15_3000HSINPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_FlatPt_15_3000HS/CMSSW_6_2_0_pre8-PRE_ST62_V8-v1/GEN-SIM',location='STD')}
#the following dataset used to be in input but is currently not valid das datasets
steps['QCD_FlatPt_15_3000HS__DIGIPU1INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_FlatPt_15_3000/CMSSW_5_2_2-PU_START52_V4_special_120326-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',location='STD')}
steps['TTbar__DIGIPU1INPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/CMSSW_5_2_2-PU_START52_V4_special_120326-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',location='STD')}

# 13 TeV recycle GEN-SIM input
steps['MinBias_13INPUT']={'INPUT':InputInfo(dataSet='/RelValMinBias_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['QCD_Pt_3000_3500_13INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_3000_3500_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['QCD_Pt_600_800_13INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_600_800_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['QCD_Pt_80_120_13INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_80_120_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['TTbar_13INPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['TTbarLepton_13INPUT']={'INPUT':InputInfo(dataSet='/RelValTTbarLepton_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['ZEE_13INPUT']={'INPUT':InputInfo(dataSet='/RelValZEE_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['Wjet_Pt_80_120_13INPUT']={'INPUT':InputInfo(dataSet='/RelValWjet_Pt_80_120_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['Wjet_Pt_3000_3500_13INPUT']={'INPUT':InputInfo(dataSet='/RelValWjet_Pt_3000_3500_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['LM1_sfts_13INPUT']={'INPUT':InputInfo(dataSet='/RelValLM1_sfts_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['QCD_FlatPt_15_3000_13INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_FlatPt_15_3000_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['QCD_FlatPt_15_3000HS_13INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_FlatPt_15_3000HS_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['ZpMM_2250_13TeV_TauolaINPUT']={'INPUT':InputInfo(dataSet='/RelValZpMM_2250_13TeV_Tauola/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['ZpEE_2250_13TeV_TauolaINPUT']={'INPUT':InputInfo(dataSet='/RelValZpEE_2250_13TeV_Tauola/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['ZpTT_1500_13TeV_TauolaINPUT']={'INPUT':InputInfo(dataSet='/RelValZpTT_1500_13TeV_Tauola/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['MinBiasHS_13INPUT']={'INPUT':InputInfo(dataSet='/RelValMinBiasHS_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['Higgs200ChargedTaus_13INPUT']={'INPUT':InputInfo(dataSet='/RelValHiggs200ChargedTaus_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['JpsiMM_13INPUT']={'INPUT':InputInfo(dataSet='/RelValJpsiMM_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['WE_13INPUT']={'INPUT':InputInfo(dataSet='/RelValWE_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['WM_13INPUT']={'INPUT':InputInfo(dataSet='/RelValWM_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['WpM_13INPUT']={'INPUT':InputInfo(dataSet='/RelValWpM_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['ZMM_13INPUT']={'INPUT':InputInfo(dataSet='/RelValZMM_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['ZpMM_13INPUT']={'INPUT':InputInfo(dataSet='/RelValZpMM_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['ZTT_13INPUT']={'INPUT':InputInfo(dataSet='/RelValZTT_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['H130GGgluonfusion_13INPUT']={'INPUT':InputInfo(dataSet='/RelValH130GGgluonfusion_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['PhotonJets_Pt_10_13INPUT']={'INPUT':InputInfo(dataSet='/RelValPhotonJets_Pt_10_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['QQH1352T_Tauola_13INPUT']={'INPUT':InputInfo(dataSet='/RelValQQH1352T_Tauola_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['ADDMonoJet_d3MD3_13INPUT']={'INPUT':InputInfo(dataSet='/RelValADDMonoJet_d3MD3_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['BeamHalo_13INPUT']={'INPUT':InputInfo(dataSet='/RelValBeamHalo_13/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
# particle guns with postLS1 geometry recycle GEN-SIM input
steps['SingleElectronPt10_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleElectronPt10_UP15/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['SingleElectronPt35_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleElectronPt35_UP15/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['SingleElectronPt1000_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleElectronPt1000_UP15/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['SingleElectronFlatPt1To100_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleElectronFlatPt1To100_UP15/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['SingleGammaPt10_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleGammaPt10_UP15/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['SingleGammaPt35_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleGammaPt35_UP15/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['SingleMuPt1_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt1_UP15/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['SingleMuPt10_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt10_UP15/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['SingleMuPt100_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt100_UP15/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}
steps['SingleMuPt1000_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt1000_UP15/%s/GEN-SIM'%(baseDataSetRelease[6],),location='STD')}

#input for fast sim workflows to be added - TODO


## high stat step1
ecalHcal={
    '-s':'GEN,SIM,DIGI,DIGI2RAW,RAW2DIGI,L1Reco,RECO,EI',
    '--datatier':'GEN-SIM-DIGI-RAW-RECO',
    #'--geometry':'ECALHCAL',
    '--eventcontent':'FEVTDEBUG',
    '--customise':'Validation/Configuration/ECALHCAL.customise,SimGeneral/MixingModule/fullMixCustomize_cff.setCrossingFrameOn',
    '--beamspot':'NoSmear'}

steps['SingleElectronE120EHCAL']=merge([{'cfg':'SingleElectronE120EHCAL_cfi'},ecalHcal,Kby(25,250),step1Defaults])
steps['SinglePiE50HCAL']=merge([{'cfg':'SinglePiE50HCAL_cfi'},ecalHcal,Kby(25,250),step1Defaults])

steps['MinBiasHS']=gen('MinBias_8TeV_cfi',Kby(25,300))
steps['MinBiasHS_13']=gen2015('MinBias_13TeV_cfi',Kby(25,300))
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
steps['Higgs200ChargedTaus_13']=gen2015('H200ChargedTaus_Tauola_13TeV_cfi',Kby(9,100))
steps['JpsiMM_13']=gen2015('JpsiMM_13TeV_cfi',Kby(66,1000))
steps['WE_13']=gen2015('WE_13TeV_cfi',Kby(9,100))
steps['WM_13']=gen2015('WM_13TeV_cfi',Kby(9,200))
steps['WpM_13']=gen2015('WpM_13TeV_cfi',Kby(9,200))
steps['ZMM_13']=gen2015('ZMM_13TeV_cfi',Kby(18,300))
steps['ZpMM_13']=gen2015('ZpMM_13TeV_cfi',Kby(9,200))

steps['ZTT']=genS('ZTT_Tauola_All_hadronic_8TeV_cfi',Kby(9,150))
steps['H130GGgluonfusion']=genS('H130GGgluonfusion_8TeV_cfi',Kby(9,100))
steps['PhotonJets_Pt_10']=genS('PhotonJet_Pt_10_8TeV_cfi',Kby(9,150))
steps['QQH1352T_Tauola']=genS('QQH1352T_Tauola_8TeV_cfi',Kby(9,100))
steps['ZTT_13']=gen2015('ZTT_Tauola_All_hadronic_13TeV_cfi',Kby(9,150))
steps['H130GGgluonfusion_13']=gen2015('H130GGgluonfusion_13TeV_cfi',Kby(9,100))
steps['PhotonJets_Pt_10_13']=gen2015('PhotonJet_Pt_10_13TeV_cfi',Kby(9,150))
steps['QQH1352T_Tauola_13']=gen2015('QQH1352T_Tauola_13TeV_cfi',Kby(9,100))
steps['ZmumuJets_Pt_20_300']=gen('ZmumuJets_Pt_20_300_GEN_8TeV_cfg',Kby(25,100))
steps['ADDMonoJet_d3MD3']=genS('ADDMonoJet_8TeV_d3MD3_cfi',Kby(9,100))
steps['ADDMonoJet_d3MD3_13']=gen2015('ADDMonoJet_13TeV_d3MD3_cfi',Kby(9,100))

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
steps['WpMINPUT']={'INPUT':InputInfo(dataSet='/RelValWpM/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['ZpMMINPUT']={'INPUT':InputInfo(dataSet='/RelValZpMM/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['ZpMM_2250_8TeV_TauolaINPUT']={'INPUT':InputInfo(dataSet='/RelValZpMM_2250_8TeV_Tauola/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['ZpEE_2250_8TeV_TauolaINPUT']={'INPUT':InputInfo(dataSet='/RelValZpEE_2250_8TeV_Tauola/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['ZpTT_1500_8TeV_TauolaINPUT']={'INPUT':InputInfo(dataSet='/RelValZpTT_1500_8TeV_Tauola/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}


steps['ZmumuJets_Pt_20_300INPUT']={'INPUT':InputInfo(dataSet='/RelValZmumuJets_Pt_20_300/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}


steps['Cosmics']=merge([{'cfg':'UndergroundCosmicMu_cfi.py','--scenario':'cosmics'},Kby(666,100000),step1Defaults])
steps['BeamHalo']=merge([{'cfg':'BeamHalo_cfi.py','--scenario':'cosmics'},Kby(9,100),step1Defaults])
steps['BeamHalo_13']=merge([{'cfg':'BeamHalo_13TeV_cfi.py','--scenario':'cosmics'},Kby(9,100),step1Up2015Defaults])

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


   
## pPb tests
step1PPbDefaults={'--beamspot':'Realistic8TeVCollisionPPbBoost'}
steps['AMPT_PPb_5020GeV_MinimumBias']=merge([{'-n':10},step1PPbDefaults,genS('AMPT_PPb_5020GeV_MinimumBias_cfi',Kby(9,100))])
steps['AMPT_PPb_5020GeV_MinimumBiasINPUT']={'INPUT':InputInfo(dataSet='/RelValAMPT_PPb_5020GeV_MinimumBias/%s/GEN-SIM'%(baseDataSetRelease[5],),location='STD')}

## heavy ions tests
U500by1={'--relval': '500,1'}
U80by1={'--relval': '80,1'}

hiDefaults={'--conditions':'auto:starthi_HIon',
           '--scenario':'HeavyIons'}

steps['HydjetQ_MinBias_2760GeV']=merge([{'-n':1},hiDefaults,genS('Hydjet_Quenched_MinBias_2760GeV_cfi',U500by1)])
steps['HydjetQ_MinBias_2760GeVINPUT']={'INPUT':InputInfo(dataSet='/RelValHydjetQ_MinBias_2760GeV/%s/GEN-SIM'%(baseDataSetRelease[1],),location='STD',split=5)}
steps['HydjetQ_B0_2760GeV']=merge([{'-n':1},hiDefaults,genS('Hydjet_Quenched_B0_2760GeV_cfi',U80by1)])
steps['HydjetQ_B0_2760GeVINPUT']={'INPUT':InputInfo(dataSet='/RelValHydjetQ_B0_2760GeV/%s/GEN-SIM'%(baseDataSetRelease[4],),location='STD')}
steps['HydjetQ_B3_2760GeV']=merge([{'-n':1},hiDefaults,genS('Hydjet_Quenched_B3_2760GeV_cfi',U80by1)])
steps['HydjetQ_B3_2760GeVINPUT']={'INPUT':InputInfo(dataSet='/RelValHydjetQ_B3_2760GeV/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
#steps['HydjetQ_B5_2760GeV']=merge([{'-n':1},hiDefaults,genS('Hydjet_Quenched_B5_2760GeV_cfi',U80by1)])
#steps['HydjetQ_B5_2760GeVINPUT']={'INPUT':InputInfo(dataSet='/RelValHydjetQ_B5_2760GeV/%s/GEN-SIM'%(baseDataSetRelease[],),location='STD')}
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
steps['TTbarFS_13']=merge([{'cfg':'TTbar_Tauola_13TeV_cfi'},Kby(100,1000),step1FastDefaults])
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
steps['ZEEFS_13']=merge([{'cfg':'ZEE_13TeV_cfi'},Kby(100,2000),step1FastDefaults])
steps['ZTTFS_13']=merge([{'cfg':'ZTT_Tauola_OneLepton_OtherHadrons_13TeV_cfi'},Kby(100,2000),step1FastDefaults])
steps['QCDFlatPt153000FS_13']=merge([{'cfg':'QCDForPF_13TeV_cfi'},Kby(27,2000),step1FastDefaults])
steps['QCD_Pt_80_120FS_13']=merge([{'cfg':'QCD_Pt_80_120_13TeV_cfi'},Kby(100,500),stCond,step1FastDefaults])
steps['QCD_Pt_3000_3500FS_13']=merge([{'cfg':'QCD_Pt_3000_3500_13TeV_cfi'},Kby(100,500),stCond,step1FastDefaults])
steps['H130GGgluonfusionFS_13']=merge([{'cfg':'H130GGgluonfusion_13TeV_cfi'},step1FastDefaults])
#GF: include fast_sim_13 
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

PU={'-n':10,'--pileup':'default','--pileup_input':'das:/RelValMinBias/%s/GEN-SIM'%(baseDataSetRelease[0],)}
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
                  '-s'            : 'DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval,RAW2DIGI,L1Reco',
                  '--datatier'    : 'GEN-SIM-DIGI-RAW-HLTDEBUG',
                  '--eventcontent': 'FEVTDEBUGHLT',
                  '--conditions'  : 'auto:startup',
                  }
#for 2015
step2Upg2015Defaults = {'-s'     :'DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval,RAW2DIGI,L1Reco',
                 '--conditions'  :'auto:upgradePLS1', 
                 '--magField'    : '38T_PostLS1',
                 '--datatier'    :'GEN-SIM-DIGI-RAW',
                 '-n'            :'10',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--customise'   : 'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1',
                 '--geometry'    : 'Extended2015'
                  }
steps['DIGIUP15']=merge([step2Upg2015Defaults]) # todo: remove UP from label

steps['DIGIPROD1']=merge([{'-s':'DIGI,L1,DIGI2RAW,HLT:@relval,RAW2DIGI,L1Reco','--eventcontent':'RAWSIM','--datatier':'GEN-SIM-RAW'},step2Defaults])
steps['DIGI']=merge([step2Defaults])
#steps['DIGI2']=merge([stCond,step2Defaults])
steps['DIGICOS']=merge([{'--scenario':'cosmics','--eventcontent':'FEVTDEBUG','--datatier':'GEN-SIM-DIGI-RAW'},stCond,step2Defaults])
steps['DIGIHAL']=merge([{'--scenario':'cosmics','--eventcontent':'FEVTDEBUG','--datatier':'GEN-SIM-DIGI-RAW','--magField':'38T_PostLS1'},step2Upg2015Defaults])

steps['DIGIPU1']=merge([PU,step2Defaults])
steps['REDIGIPU']=merge([{'-s':'reGEN,reDIGI,L1,DIGI2RAW,HLT:@relval,RAW2DIGI,L1Reco'},steps['DIGIPU1']])

steps['DIGI_ID']=merge([{'--restoreRND':'HLT','--process':'HLT2'},steps['DIGI']])

steps['RESIM']=merge([{'-s':'reGEN,reSIM','-n':10},steps['DIGI']])
steps['RESIMDIGI']=merge([{'-s':'reGEN,reSIM,DIGI,L1,DIGI2RAW,HLT:@relval,RAW2DIGI,L1Reco','-n':10,'--restoreRNDSeeds':'','--process':'HLT'},steps['DIGI']])

    
steps['DIGIHI']=merge([{'--conditions':'auto:starthi_HIon', '-s':'DIGI:pdigi_valid,L1,DIGI2RAW,HLT:HIon,RAW2DIGI,L1Reco', '--inputCommands':'"keep *","drop *_simEcalPreshowerDigis_*_*"', '-n':10}, hiDefaults, step2Defaults])

#wmsplit['DIGIHI']=5


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
step4Up2015Defaults = { 
                        '-s'            : 'ALCA:TkAlMuonIsolated+TkAlMinBias+EcalCalElectron+HcalCalIsoTrk+MuAlOverlaps',
                        '-n'            : 1000,
                        '--conditions'  : 'auto:upgradePLS1',
                        '--datatier'    : 'ALCARECO',
                        '--eventcontent': 'ALCARECO',
                  }

steps['RERECOPU']=steps['RERECOPU1']

steps['ALCATT']=merge([{'--filein':'file:step3.root'},step4Defaults])
steps['ALCAMIN']=merge([{'-s':'ALCA:TkAlMinBias','--filein':'file:step3.root'},stCond,step4Defaults])
steps['ALCACOS']=merge([{'-s':'ALCA:TkAlCosmics0T+MuAlGlobalCosmics+HcalCalHOCosmics'},stCond,step4Defaults])
steps['ALCABH']=merge([{'-s':'ALCA:TkAlBeamHalo+MuAlBeamHaloOverlaps+MuAlBeamHalo'},stCond,step4Defaults])
steps['ALCAHAL']=merge([{'-s':'ALCA:TkAlBeamHalo+MuAlBeamHaloOverlaps+MuAlBeamHalo'},step4Up2015Defaults])
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
steps['HARVESTHAL']={'-s'          :'HARVESTING:dqmHarvesting',
                     '--conditions':'auto:upgradePLS1',
                     '--magField'  : '38T_PostLS1',
                     '--mc'        :'',
                     '--filein'    :'file:step3_inDQM.root',
                     #'--filein'    :'file:step3_inAODSIM.root',
                     '--scenario'  :'cosmics',
                     '--customise' : 'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1'}
steps['HARVESTFS']={'-s':'HARVESTING:validationHarvestingFS',
                   '--conditions':'auto:startup',
                   '--mc':'',
                   '--scenario':'pp'}
steps['HARVESTHI']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                   '--conditions':'auto:starthi_HIon',
                   '--mc':'',
                   '--scenario':'HeavyIons'}

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
                '--secondfilein':'filelist:step1_dasquery.log'}

steps['SKIMDreHLT'] = merge([ {'--conditions':'auto:com10_%s'%menu,'--filein':'file:step3.root'}, steps['SKIMD'] ])

steps['SKIMCOSD']={'-s':'SKIM:all',
                   '--conditions':'auto:com10',
                   '--data':'',
                   '--scenario':'cosmics',
                   '--filein':'file:step2.root',
                   '--secondfilein':'filelist:step1_dasquery.log'}
                 

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
                                  '--secondfilein':'filelist:step1_dasquery.log',
                                  },
                                 steps['RECOD']])


steps['COPYPASTE']={'-s':'NONE',
                    '--conditions':'auto:startup',
                    '--output':'\'[{"t":"RAW","e":"ALL"}]\'',
                    '--customise_commands':'"process.ALLRAWoutput.fastCloning=cms.untracked.bool(False)"'}



# You will need separate scenarios HERE for full and fast. DON'T CHANGE THE ORDER, only
# append new keys. Otherwise the numbering for the runTheMatrix tests will change.
upgradeKeys=['2017',
             '2019',
             'BE5D',
             '2017Fast',
             'BE5DFast',
             'BE5DForwardFast',
             '2019WithGEM',
             'BE5DPixel10D',
             '2017Aging',
             '2019Aging',
             'Extended2023',
             'Extended2023HGCalMuon',
             'Extended2023SHCal',
             'Extended2023SHCal4Eta',
             'Extended2023TTI',
             'Extended2023Muon',
             'Extended2023CFCal',
             'Extended2023CFCal4Eta',
             'Extended2023Pixel',            
             'Extended2023SHCalNoTaper',
             'Extended2023SHCalNoTaper4Eta',
             'Extended2023HGCal',
             'Extended2023HGCalMuon4Eta',
             'Extended2023Muon4Eta'
	     ]
upgradeGeoms={ '2017' : 'Extended2017',
               '2019' : 'Extended2019',
               '2019WithGEM' : 'Extended2019',
               '2017Aging' : 'Extended2017',
               '2019Aging' : 'Extended2019',
               'BE5D' : 'ExtendedPhase2TkBE5D',
               'BE5DPixel10D' : 'ExtendedPhase2TkBE5DPixel10D',
               '2017Fast' : 'Extended2017',
               'BE5DFast' : 'ExtendedPhase2TkBE',
               'BE5DForwardFast' : 'ExtendedPhase2TkBEForward',
               'Extended2023' : 'Extended2023,Extended2023Reco',
               'Extended2023HGCalMuon' : 'Extended2023HGCalMuon,Extended2023HGCalMuonReco',
               'Extended2023SHCal' : 'Extended2023SHCal,Extended2023SHCalReco',
               'Extended2023SHCal4Eta' : 'Extended2023SHCal4Eta,Extended2023SHCalReco',
               'Extended2023TTI' : 'Extended2023TTI,Extended2023TTIReco',
               'Extended2023Muon' : 'Extended2023Muon,Extended2023MuonReco',
               'Extended2023Muon4Eta' : 'Extended2023Muon4Eta,Extended2023Muon4EtaReco',
               'Extended2023CFCal' : 'Extended2023CFCal,Extended2023CFCalReco',
               'Extended2023CFCal4Eta' : 'Extended2023CFCal4Eta,Extended2023CFCal4EtaReco',
               'Extended2023Pixel' : 'Extended2023Pixel,Extended2023PixelReco',
               'Extended2023SHCalNoTaper' : 'Extended2023SHCalNoTaper,Extended2023SHCalNoTaperReco',
               'Extended2023SHCalNoTaper4Eta' : 'Extended2023SHCalNoTaper4Eta,Extended2023SHCalNoTaper4EtaReco',
               'Extended2023HGCal' : 'Extended2023HGCal,Extended2023HGCalReco',
               'Extended2023HGCalMuon4Eta' : 'Extended2023HGCalMuon4Eta,Extended2023HGCalMuon4EtaReco'
               }
upgradeGTs={ '2017' : 'auto:upgrade2017',
             '2019' : 'auto:upgrade2019',
             '2019WithGEM' : 'auto:upgrade2019',
             '2017Aging' : 'W17_300_62E2::All',
             '2019Aging' : 'W19_300_62E2::All',
             'BE5D' : 'auto:upgradePLS3',
             'BE5DPixel10D' : 'auto:upgradePLS3',
             '2017Fast' : 'auto:upgrade2017',
             'BE5DFast' : 'auto:upgrade2019',
             'BE5DForwardFast' : 'auto:upgrade2019',
             'Extended2023' : 'auto:upgradePLS3',
             'Extended2023HGCalMuon' : 'auto:upgradePLS3',
             'Extended2023SHCal' : 'auto:upgradePLS3',
             'Extended2023SHCal4Eta' : 'auto:upgradePLS3',
             'Extended2023TTI' : 'auto:upgradePLS3',
             'Extended2023Muon' : 'auto:upgradePLS3',
             'Extended2023Muon4Eta' : 'auto:upgradePLS3',
             'Extended2023CFCal' : 'auto:upgradePLS3',
             'Extended2023CFCal4Eta' : 'auto:upgradePLS3',
             'Extended2023Pixel' : 'auto:upgradePLS3',
             'Extended2023SHCalNoTaper' : 'auto:upgradePLS3',
             'Extended2023SHCalNoTaper4Eta' : 'auto:upgradePLS3',
             'Extended2023HGCal' : 'auto:upgradePLS3',
             'Extended2023HGCalMuon4Eta' : 'auto:upgradePLS3'
             }
upgradeCustoms={ '2017' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017',
                 '2019' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019',
                 '2019WithGEM' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019WithGem',
                 '2017Aging' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2017,SLHCUpgradeSimulations/Configuration/aging.customise_aging_300',
                 '2019Aging' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2019,SLHCUpgradeSimulations/Configuration/aging.customise_aging_300',
                 'BE5D' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_phase2_BE5D',
                 'BE5DPixel10D' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_phase2_BE5DPixel10D',
                 '2017Fast' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.fastsimDefault',
                 'BE5DFast' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.fastsimPhase2',
                 'BE5DForwardFast' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.fastsimPhase2',
                 'Extended2023' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023',
                 'Extended2023HGCalMuon' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023',
                 'Extended2023SHCal' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023',
                 'Extended2023SHCal4Eta' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023',
                 'Extended2023TTI' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023TTI',
                 'Extended2023Muon' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023Muon',
                 'Extended2023Muon4Eta' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023Muon',
                 'Extended2023CFCal' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023',
                 'Extended2023CFCal4Eta' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023',
                 'Extended2023Pixel' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023Pixel',
                 'Extended2023SHCalNoTaper' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023',
                 'Extended2023SHCalNoTaper4Eta' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023',
                 'Extended2023HGCal' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023',
                 'Extended2023HGCalMuon4Eta' : 'SLHCUpgradeSimulations/Configuration/combinedCustoms.cust_2023'
                 }
### remember that you need to add a new step for phase 2 to include the track trigger
### remember that you need to add fastsim

# step1 is normal gen-sim
# step2 is digi
# step3 is reco
# step4 is harvest
# step5 is digi+l1tracktrigger
# step6 is fastsim
# step7 is fastsim harvesting

upgradeSteps=['GenSimFull','GenSimHLBeamSpotFull','DigiFull','RecoFull','HARVESTFull','DigiTrkTrigFull','FastSim','HARVESTFast']

upgradeScenToRun={ '2017':['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
                   '2019':['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
                   '2019WithGEM':['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
                   '2017Aging':['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
                   '2019Aging':['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
                   'BE5D':['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
                   'BE5DPixel10D':['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
                   '2017Fast':['FastSim','HARVESTFast'],
                   'BE5DFast':['FastSim','HARVESTFast'],
                   'BE5DForwardFast':['FastSim','HARVESTFast'],
                   'Extended2023':['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
                   'Extended2023HGCalMuon':['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
                   'Extended2023SHCal':['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
                   'Extended2023SHCal4Eta':['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
                   'Extended2023TTI':['GenSimHLBeamSpotFull','DigiTrkTrigFull'], ##no need to go beyond local reco
                   'Extended2023Muon':['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
                   'Extended2023Muon4Eta':['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
                   'Extended2023CFCal':['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
                   'Extended2023CFCal4Eta':['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
                   'Extended2023Pixel':['GenSimFull','DigiFull','RecoFull','HARVESTFull'],          
                   'Extended2023SHCalNoTaper':['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
                   'Extended2023SHCalNoTaper4Eta':['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
                   'Extended2023HGCal':['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
                   'Extended2023HGCalMuon4Eta':['GenSimFull','DigiFull','RecoFull','HARVESTFull']
                   }

upgradeStepDict={}
for step in upgradeSteps:
    upgradeStepDict[step]={}

# just make all combinations - yes, some will be nonsense.. but then these are not used unless
# specified above
for k in upgradeKeys:
    upgradeStepDict['GenSimFull'][k]= {'-s' : 'GEN,SIM',
                                       '-n' : 10,
                                       '--conditions' : upgradeGTs[k],
                                       '--beamspot' : 'Gauss',
                                       '--magField' : '38T_PostLS1',
                                       '--datatier' : 'GEN-SIM',
                                       '--eventcontent': 'FEVTDEBUG',
                                       '--geometry' : upgradeGeoms[k]
                                       }
    if upgradeCustoms[k]!=None : upgradeStepDict['GenSimFull'][k]['--customise']=upgradeCustoms[k]
    
    upgradeStepDict['GenSimHLBeamSpotFull'][k]= {'-s' : 'GEN,SIM',
                                       '-n' : 10,
                                       '--conditions' : upgradeGTs[k],
                                       '--beamspot' : 'HLLHC',
                                       '--magField' : '38T_PostLS1',
                                       '--datatier' : 'GEN-SIM',
                                       '--eventcontent': 'FEVTDEBUG',
                                       '--geometry' : upgradeGeoms[k]
                                       }
    if upgradeCustoms[k]!=None : upgradeStepDict['GenSimHLBeamSpotFull'][k]['--customise']=upgradeCustoms[k]

    upgradeStepDict['DigiFull'][k] = {'-s':'DIGI:pdigi_valid,L1,DIGI2RAW',
                                      '--conditions':upgradeGTs[k],
                                      '--datatier':'GEN-SIM-DIGI-RAW',
                                      '-n':'10',
                                      '--magField' : '38T_PostLS1',
                                      '--eventcontent':'FEVTDEBUGHLT',
                                      '--geometry' : upgradeGeoms[k]
                                      }
    if upgradeCustoms[k]!=None : upgradeStepDict['DigiFull'][k]['--customise']=upgradeCustoms[k]

    upgradeStepDict['DigiTrkTrigFull'][k] = {'-s':'DIGI:pdigi_valid,L1,L1TrackTrigger,DIGI2RAW,RECO:pixeltrackerlocalreco',
                                             '--conditions':upgradeGTs[k],
                                             '--datatier':'GEN-SIM-DIGI-RAW',
                                             '-n':'10',
                                             '--magField' : '38T_PostLS1',
                                             '--eventcontent':'FEVTDEBUGHLT',
                                             '--geometry' : upgradeGeoms[k]
                                             }
    if upgradeCustoms[k]!=None : upgradeStepDict['DigiTrkTrigFull'][k]['--customise']=upgradeCustoms[k]

    upgradeStepDict['RecoFull'][k] = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                                      '--conditions':upgradeGTs[k],
                                      '--datatier':'GEN-SIM-RECO,DQM',
                                      '-n':'10',
                                      '--eventcontent':'FEVTDEBUGHLT,DQM',
                                      '--magField' : '38T_PostLS1',
                                      '--geometry' : upgradeGeoms[k]
                                      }
    if upgradeCustoms[k]!=None : upgradeStepDict['RecoFull'][k]['--customise']=upgradeCustoms[k]
    
    upgradeStepDict['HARVESTFull'][k]={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                                    '--conditions':upgradeGTs[k],
                                    '--mc':'',
                                    '--magField' : '38T_PostLS1',
                                    '--geometry' : upgradeGeoms[k],
                                    '--scenario' : 'pp'
                                    }
    if upgradeCustoms[k]!=None : upgradeStepDict['HARVESTFull'][k]['--customise']=upgradeCustoms[k]

    upgradeStepDict['FastSim'][k]={'-s':'GEN,SIM,RECO,VALIDATION',
                                   '--eventcontent':'FEVTDEBUGHLT,DQM',
                                   '--datatier':'GEN-SIM-DIGI-RECO,DQM',
                                   '--conditions':upgradeGTs[k],
                                   '--fast':'',
                                   '--geometry' : upgradeGeoms[k],
                                   '--relval':'27000,3000'}
    if upgradeCustoms[k]!=None : upgradeStepDict['FastSim'][k]['--customise']=upgradeCustoms[k]

    upgradeStepDict['HARVESTFast'][k]={'-s':'HARVESTING:validationHarvestingFS',
                                    '--conditions':upgradeGTs[k],
                                    '--mc':'',
                                    '--magField' : '38T_PostLS1',
                                    '--geometry' : upgradeGeoms[k],
                                    '--scenario' : 'pp'
                                    }
    if upgradeCustoms[k]!=None : upgradeStepDict['HARVESTFast'][k]['--customise']=upgradeCustoms[k]


upgradeFragments=['FourMuPt_1_200_cfi','SingleElectronPt10_cfi',
                  'SingleElectronPt35_cfi','SingleElectronPt1000_cfi',
                  'SingleGammaPt10_cfi','SingleGammaPt35_cfi','SingleMuPt1_cfi','SingleMuPt10_cfi',
                  'SingleMuPt100_cfi','SingleMuPt1000_cfi','TTbarLepton_Tauola_8TeV_cfi','Wjet_Pt_80_120_8TeV_cfi',
                  'Wjet_Pt_3000_3500_8TeV_cfi','LM1_sfts_8TeV_cfi','QCD_Pt_3000_3500_8TeV_cfi','QCD_Pt_600_800_8TeV_cfi',
                  'QCD_Pt_80_120_8TeV_cfi','H200ChargedTaus_Tauola_8TeV_cfi','JpsiMM_8TeV_cfi','TTbar_Tauola_8TeV_cfi',
                  'WE_8TeV_cfi','ZEE_8TeV_cfi','ZTT_Tauola_All_hadronic_8TeV_cfi','H130GGgluonfusion_8TeV_cfi',
                  'PhotonJet_Pt_10_8TeV_cfi','QQH1352T_Tauola_8TeV_cfi','MinBias_TuneZ2star_8TeV_pythia6_cff','WM_8TeV_cfi',
                  'ZMM_8TeV_cfi','ADDMonoJet_8TeV_d3MD3_cfi','ZpMM_8TeV_cfi','WpM_8TeV_cfi',
                  'Wjet_Pt_80_120_14TeV_cfi','Wjet_Pt_3000_3500_14TeV_cfi','LM1_sfts_14TeV_cfi','QCD_Pt_3000_3500_14TeV_cfi',
                  'QCD_Pt_80_120_14TeV_cfi','H200ChargedTaus_Tauola_14TeV_cfi','JpsiMM_14TeV_cfi','TTbar_Tauola_14TeV_cfi',
                  'WE_14TeV_cfi','ZEE_14TeV_cfi','ZTT_Tauola_All_hadronic_14TeV_cfi','H130GGgluonfusion_14TeV_cfi',
                  'PhotonJet_Pt_10_14TeV_cfi','QQH1352T_Tauola_14TeV_cfi',
                  'MinBias_TuneZ2star_14TeV_pythia6_cff','WM_14TeV_cfi','ZMM_14TeV_cfi',
		  'FourMuExtendedPt_1_200_cfi',
		  'TenMuExtendedE_0_200_cfi',
		  'SingleElectronPt10Extended_cfi',
                  'SingleElectronPt35Extended_cfi',
		  'SingleElectronPt1000Extended_cfi',
                  'SingleGammaPt10Extended_cfi',
		  'SingleGammaPt35Extended_cfi',
		  'SingleMuPt1Extended_cfi',
		  'SingleMuPt10Extended_cfi',
                  'SingleMuPt100Extended_cfi',
		  'SingleMuPt1000Extended_cfi','TenMuE_0_200_cfi']

howMuches={'FourMuPt_1_200_cfi':Kby(10,100),
           'TenMuE_0_200_cfi':Kby(10,100),
           'FourMuExtendedPt_1_200_cfi':Kby(10,100),
           'TenMuExtendedE_0_200_cfi':Kby(10,100),
           'SingleElectronPt10_cfi':Kby(9,300),
           'SingleElectronPt35_cfi':Kby(9,500),
           'SingleElectronPt1000_cfi':Kby(9,50),
           'SingleGammaPt10_cfi':Kby(9,300),
           'SingleGammaPt35_cfi':Kby(9,50),
           'SingleMuPt1_cfi':Kby(25,1000),
           'SingleMuPt10_cfi':Kby(25,500),
           'SingleMuPt100_cfi':Kby(9,500),
           'SingleMuPt1000_cfi':Kby(9,500),
           'SingleElectronPt10Extended_cfi':Kby(9,300),
           'SingleElectronPt35Extended_cfi':Kby(9,500),
           'SingleElectronPt1000Extended_cfi':Kby(9,50),
           'SingleGammaPt10Extended_cfi':Kby(9,300),
           'SingleGammaPt35Extended_cfi':Kby(9,50),
           'SingleMuPt1Extended_cfi':Kby(25,1000),
           'SingleMuPt10Extended_cfi':Kby(25,500),
           'SingleMuPt100Extended_cfi':Kby(9,500),
           'SingleMuPt1000Extended_cfi':Kby(9,500),
           'TTbarLepton_Tauola_8TeV_cfi':Kby(9,100),
           'Wjet_Pt_80_120_8TeV_cfi':Kby(9,100),
           'Wjet_Pt_3000_3500_8TeV_cfi':Kby(9,50),
           'LM1_sfts_8TeV_cfi':Kby(9,100),
           'QCD_Pt_3000_3500_8TeV_cfi':Kby(9,25),
           'QCD_Pt_600_800_8TeV_cfi':Kby(9,50),
           'QCD_Pt_80_120_8TeV_cfi':Kby(9,100),
           'H200ChargedTaus_Tauola_8TeV_cfi':Kby(9,100),
           'JpsiMM_8TeV_cfi':Kby(66,1000),
           'TTbar_Tauola_8TeV_cfi':Kby(9,100),
           'WE_8TeV_cfi':Kby(9,100),
           'ZEE_8TeV_cfi':Kby(9,100),
           'ZTT_Tauola_All_hadronic_8TeV_cfi':Kby(9,15),
           'H130GGgluonfusion_8TeV_cfi':Kby(9,100),
           'PhotonJet_Pt_10_8TeV_cfi':Kby(9,150),
           'QQH1352T_Tauola_8TeV_cfi':Kby(9,100),
           'MinBias_TuneZ2star_8TeV_pythia6_cff':Kby(9,30),
           'WM_8TeV_cfi':Kby(9,200),
           'ZMM_8TeV_cfi':Kby(18,300),
           'ADDMonoJet_8TeV_d3MD3_cfi':Kby(9,100),
           'ZpMM_8TeV_cfi':Kby(9,200),
           'WpM_8TeV_cfi':Kby(9,200),
           'Wjet_Pt_80_120_14TeV_cfi':Kby(9,100),
           'Wjet_Pt_3000_3500_14TeV_cfi':Kby(9,50),
           'LM1_sfts_14TeV_cfi':Kby(9,100),
           'QCD_Pt_3000_3500_14TeV_cfi':Kby(9,25),
           'QCD_Pt_80_120_14TeV_cfi':Kby(9,100),
           'H200ChargedTaus_Tauola_14TeV_cfi':Kby(9,100),
           'JpsiMM_14TeV_cfi':Kby(66,1000),
           'TTbar_Tauola_14TeV_cfi':Kby(9,100),
           'WE_14TeV_cfi':Kby(9,100),
           'ZEE_14TeV_cfi':Kby(9,100),
           'ZTT_Tauola_All_hadronic_14TeV_cfi':Kby(9,150),
           'H130GGgluonfusion_14TeV_cfi':Kby(9,100),
           'PhotonJet_Pt_10_14TeV_cfi':Kby(9,150),
           'QQH1352T_Tauola_14TeV_cfi':Kby(9,100),
           'MinBias_TuneZ2star_14TeV_pythia6_cff':Kby(9,300),
           'WM_14TeV_cfi':Kby(9,200),
           'ZMM_14TeV_cfi':Kby(18,300)
           }

upgradeDatasetFromFragment={'FourMuPt_1_200_cfi': 'FourMuPt1_200',
                            'FourMuExtendedPt_1_200_cfi': 'FourMuExtendedPt1_200',
                            'TenMuE_0_200_cfi': 'TenMuE_0_200',
                            'TenMuExtendedE_0_200_cfi': 'TenMuExtendedE_0_200',
                            'SingleElectronPt10_cfi' : 'SingleElectronPt10',
                            'SingleElectronPt35_cfi' : 'SingleElectronPt35',
                            'SingleElectronPt1000_cfi' : 'SingleElectronPt1000',
                            'SingleGammaPt10_cfi' : 'SingleGammaPt10',
                            'SingleGammaPt35_cfi' : 'SingleGammaPt35',
                            'SingleMuPt1_cfi' : 'SingleMuPt1',
                            'SingleMuPt10_cfi' : 'SingleMuPt10',
                            'SingleMuPt100_cfi' : 'SingleMuPt100',
                            'SingleMuPt1000_cfi' : 'SingleMuPt1000',
                            'SingleElectronPt10Extended_cfi' : 'SingleElectronPt10Extended',
                            'SingleElectronPt35Extended_cfi' : 'SingleElectronPt35Extended',
                            'SingleElectronPt1000Extended_cfi' : 'SingleElectronPt1000Extended',
                            'SingleGammaPt10Extended_cfi' : 'SingleGammaPt10Extended',
                            'SingleGammaPt35Extended_cfi' : 'SingleGammaPt35Extended',
                            'SingleMuPt1Extended_cfi' : 'SingleMuPt1Extended',
                            'SingleMuPt10Extended_cfi' : 'SingleMuPt10Extended',
                            'SingleMuPt100Extended_cfi' : 'SingleMuPt100Extended',
                            'SingleMuPt1000Extended_cfi' : 'SingleMuPt1000Extended',
                            'TTbarLepton_Tauola_8TeV_cfi' : 'TTbarLepton_8TeV',
                            'Wjet_Pt_80_120_8TeV_cfi' : 'Wjet_Pt_80_120_8TeV',
                            'Wjet_Pt_3000_3500_8TeV_cfi' : 'Wjet_Pt_3000_3500_8TeV',
                            'LM1_sfts_8TeV_cfi' : 'LM1_sfts_8TeV',
                            'QCD_Pt_3000_3500_8TeV_cfi' : 'QCD_Pt_3000_3500_8TeV',
                            'QCD_Pt_600_800_8TeV_cfi' : 'QCD_Pt_600_800_8TeV',
                            'QCD_Pt_80_120_8TeV_cfi' : 'QCD_Pt_80_120_8TeV',
                            'H200ChargedTaus_Tauola_8TeV_cfi' : 'Higgs200ChargedTaus_8TeV',
                            'JpsiMM_8TeV_cfi' : 'JpsiMM_8TeV',
                            'TTbar_Tauola_8TeV_cfi' : 'TTbar_8TeV',
                            'WE_8TeV_cfi' : 'WE_8TeV',
                            'ZEE_8TeV_cfi' : 'ZEE_8TeV',
                            'ZTT_Tauola_All_hadronic_8TeV_cfi' : 'ZTT_8TeV',
                            'H130GGgluonfusion_8TeV_cfi' : 'H130GGgluonfusion_8TeV',
                            'PhotonJet_Pt_10_8TeV_cfi' : 'PhotonJets_Pt_10_8TeV',
                            'QQH1352T_Tauola_8TeV_cfi' : 'QQH1352T_Tauola_8TeV',
                            'MinBias_TuneZ2star_8TeV_pythia6_cff': 'MinBias_TuneZ2star_8TeV',
                            'WM_8TeV_cfi' : 'WM_8TeV',
                            'ZMM_8TeV_cfi' : 'ZMM_8TeV',
                            'ADDMonoJet_8TeV_d3MD3_cfi' : 'ADDMonoJet_d3MD3_8TeV',
                            'ZpMM_8TeV_cfi' : 'ZpMM_8TeV',
                            'WpM_8TeV_cfi' : 'WpM_8TeV',
                            'Wjet_Pt_80_120_14TeV_cfi' : 'Wjet_Pt_80_120_14TeV',
                            'Wjet_Pt_3000_3500_14TeV_cfi' : 'Wjet_Pt_3000_3500_14TeV',
                            'LM1_sfts_14TeV_cfi' : 'LM1_sfts_14TeV',
                            'QCD_Pt_3000_3500_14TeV_cfi' : 'QCD_Pt_3000_3500_14TeV',
                            'QCD_Pt_80_120_14TeV_cfi' : 'QCD_Pt_80_120_14TeV',
                            'H200ChargedTaus_Tauola_14TeV_cfi' : 'Higgs200ChargedTaus_14TeV',
                            'JpsiMM_14TeV_cfi' : 'JpsiMM_14TeV',
                            'TTbar_Tauola_14TeV_cfi' : 'TTbar_14TeV',
                            'WE_14TeV_cfi' : 'WE_14TeV',
                            'ZEE_14TeV_cfi' : 'ZEE_14TeV',
                            'ZTT_Tauola_All_hadronic_14TeV_cfi' : 'ZTT_14TeV',
                            'H130GGgluonfusion_14TeV_cfi' : 'H130GGgluonfusion_14TeV',
                            'PhotonJet_Pt_10_14TeV_cfi' : 'PhotonJets_Pt_10_14TeV',
                            'QQH1352T_Tauola_14TeV_cfi' : 'QQH1352T_Tauola_14TeV',
                            'MinBias_TuneZ2star_14TeV_pythia6_cff' : 'MinBias_TuneZ2star_14TeV',
                            'WM_14TeV_cfi' : 'WM_14TeV',
                            'ZMM_14TeV_cfi' : 'ZMM_14TeV'
                            }



#just do everything...

for step in upgradeSteps:
    # we need to do this for each fragment
   if 'Sim' in step:
        for frag in upgradeFragments:
            howMuch=howMuches[frag]
            for key in upgradeKeys:
                k=frag[:-4]+'_'+key+'_'+step
                steps[k]=merge([ {'cfg':frag},howMuch,upgradeStepDict[step][key]])
   else:
        for key in upgradeKeys:
            k=step+'_'+key
            steps[k]=merge([upgradeStepDict[step][key]])
            
