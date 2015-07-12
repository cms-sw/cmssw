class Matrix(dict):
    def __setitem__(self,key,value):
        if key in self:
            print "ERROR in Matrix"
            print "overwritting",key,"not allowed"
        else:
            self.update({float(key):WF(float(key),value)})

    def addOverride(self,key,override):
        self[key].addOverride(override)
            
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
        self.update({keypair[0]:value})
        
class WF(list):
    def __init__(self,n,l):
        self.extend(l)
        self.num=n
        #the actual steps of this WF
        self.steps=[]
        self.overrides={}
    def addOverride(self,overrides):
        self.overrides=overrides
        
    def interpret(self,stepsDict):
        for s in self:
            print 'steps',s,stepsDict[s]
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
        
    def das(self, das_options):
        if len(self.run) is not 0:
            command = ";".join(["das_client.py %s --query '%s'" % (das_options, query) for query in self.queries()])
            command = "({0})".format(command)
        else:
            command = "das_client.py %s --query '%s'" % (das_options, self.queries()[0])
       
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

    def queries(self):
        query_by = "block" if self.ib_block else "dataset"
        query_source = "{0}#{1}".format(self.dataSet, self.ib_block) if self.ib_block else self.dataSet
        if len(self.run) is not 0:
            return ["file {0}={1} run={2} site=T2_CH_CERN".format(query_by, query_source, query_run) for query_run in self.run]
        else:
            return ["file {0}={1} site=T2_CH_CERN".format(query_by, query_source)]

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

def remove(d,key,TELL=False):
    import copy
    e = copy.deepcopy(d)
    if TELL: print "original dict, BEF: %s"%d
    del e[key]
    if TELL: print "copy-removed dict, AFT: %s"%e
    return e

# step1 gensim: for run1
step1Defaults = {'--relval'      : None, # need to be explicitly set
                 '-s'            : 'GEN,SIM',
                 '-n'            : 10,
                 '--conditions'  : 'auto:run1_mc',
                 '--beamspot'    : 'Realistic8TeVCollision',
                 '--datatier'    : 'GEN-SIM',
                 '--eventcontent': 'RAWSIM',
                 }
# step1 gensim: for postLS1
step1Up2015Defaults = {'-s' : 'GEN,SIM',
                             '-n'            : 10,
                             '--conditions'  : 'auto:run2_mc',
                             '--beamspot'    : 'NominalCollision2015',
                             '--datatier'    : 'GEN-SIM',
                             '--eventcontent': 'FEVTDEBUG',
                             '--magField'    : '38T_PostLS1',
                             '--customise'   : 'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1'
                             }

steps = Steps()
#wmsplit = {}

#### Production test section ####
steps['ProdMinBias']=merge([{'cfg':'MinBias_8TeV_pythia8_TuneCUETP8M1_cff','--relval':'9000,300'},step1Defaults])
steps['ProdTTbar']=merge([{'cfg':'TTbar_8TeV_TuneCUETP8M1_cfi','--relval':'9000,100'},step1Defaults])
steps['ProdQCD_Pt_3000_3500']=merge([{'cfg':'QCD_Pt_3000_3500_8TeV_TuneCUETP8M1_cfi','--relval':'9000,50'},step1Defaults])

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
steps['RunHI2011']={'INPUT':InputInfo(dataSet='/HIMinBiasUPC/HIRun2011-v1/RAW',label='hi2011',run=[182124],events=10000,location='STD')}


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

stCond={'--conditions':'auto:run1_mc'}
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
steps['ProdMinBias_13']=gen2015('MinBias_13TeV_pythia8_TuneCUETP8M1_cfi',Kby(9,100))
steps['ProdTTbar_13']=gen2015('TTbar_13TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['ProdZEE_13']=gen2015('ZEE_13TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['ProdQCD_Pt_3000_3500_13']=gen2015('QCD_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi',Kby(9,100))

steps['MinBias']=gen('MinBias_8TeV_pythia8_TuneCUETP8M1_cff',Kby(9,300))
steps['QCD_Pt_3000_3500']=gen('QCD_Pt_3000_3500_8TeV_TuneCUETP8M1_cfi',Kby(9,25))
steps['QCD_Pt_600_800']=gen('QCD_Pt_600_800_8TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['QCD_Pt_80_120']=gen('QCD_Pt_80_120_8TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['MinBias_13']=gen2015('MinBias_13TeV_pythia8_TuneCUETP8M1_cfi',Kby(100,300)) # set HS to provide adequate pool for PU
steps['QCD_Pt_3000_3500_13']=gen2015('QCD_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi',Kby(9,25))
steps['QCD_Pt_600_800_13']=gen2015('QCD_Pt_600_800_13TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['QCD_Pt_80_120_13']=gen2015('QCD_Pt_80_120_13TeV_TuneCUETP8M1_cfi',Kby(9,100))

steps['QCD_Pt_30_80_BCtoE_8TeV']=gen('QCD_Pt_30_80_BCtoE_8TeV_TuneCUETP8M1_cfi',Kby(9000,100))
steps['QCD_Pt_80_170_BCtoE_8TeV']=gen('QCD_Pt_80_170_BCtoE_8TeV_TuneCUETP8M1_cfi',Kby(9000,100))
steps['SingleElectronPt10']=gen('SingleElectronPt10_pythia8_cfi',Kby(9,3000))
steps['SingleElectronPt35']=gen('SingleElectronPt35_pythia8_cfi',Kby(9,500))
steps['SingleElectronPt1000']=gen('SingleElectronPt1000_pythia8_cfi',Kby(9,50))
steps['SingleElectronFlatPt1To100']=gen('SingleElectronFlatPt1To100_pythia8_cfi',Mby(2,100))
steps['SingleGammaPt10']=gen('SingleGammaPt10_pythia8_cfi',Kby(9,3000))
steps['SingleGammaPt35']=gen('SingleGammaPt35_pythia8_cfi',Kby(9,500))
steps['SingleMuPt1']=gen('SingleMuPt1_pythia8_cfi',Kby(25,1000))
steps['SingleMuPt10']=gen('SingleMuPt10_pythia8_cfi',Kby(25,500))
steps['SingleMuPt100']=gen('SingleMuPt100_pythia8_cfi',Kby(9,500))
steps['SingleMuPt1000']=gen('SingleMuPt1000_pythia8_cfi',Kby(9,500))
steps['SingleElectronPt10_UP15']=gen2015('SingleElectronPt10_pythia8_cfi',Kby(9,3000))
steps['SingleElectronPt35_UP15']=gen2015('SingleElectronPt35_pythia8_cfi',Kby(9,500))
steps['SingleElectronPt1000_UP15']=gen2015('SingleElectronPt1000_pythia8_cfi',Kby(9,50))
steps['SingleElectronFlatPt1To100_UP15']=gen2015('SingleElectronFlatPt1To100_pythia8_cfi',Mby(2,100))
steps['SingleGammaPt10_UP15']=gen2015('SingleGammaPt10_pythia8_cfi',Kby(9,3000))
steps['SingleGammaPt35_UP15']=gen2015('SingleGammaPt35_pythia8_cfi',Kby(9,500))
steps['SingleMuPt1_UP15']=gen2015('SingleMuPt1_pythia8_cfi',Kby(25,1000))
steps['SingleMuPt10_UP15']=gen2015('SingleMuPt10_pythia8_cfi',Kby(25,500))
steps['SingleMuPt100_UP15']=gen2015('SingleMuPt100_pythia8_cfi',Kby(9,500))
steps['SingleMuPt1000_UP15']=gen2015('SingleMuPt1000_pythia8_cfi',Kby(9,500))
steps['NuGun_UP15']=gen2015('SingleNuE10_cfi.py',Kby(9,100))
steps['TTbar']=gen('TTbar_8TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['TTbarLepton']=gen('TTbarLepton_8TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['ZEE']=gen('ZEE_8TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['Wjet_Pt_80_120']=gen('Wjet_Pt_80_120_8TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['Wjet_Pt_3000_3500']=gen('Wjet_Pt_3000_3500_8TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['LM1_sfts']=gen('LM1_sfts_8TeV_cfi',Kby(9,100))
steps['QCD_FlatPt_15_3000']=gen('QCDForPF_8TeV_TuneCUETP8M1_cfi',Kby(5,100))
steps['QCD_FlatPt_15_3000HS']=gen('QCDForPF_8TeV_TuneCUETP8M1_cfi',Kby(50,100))
steps['TTbar_13']=gen2015('TTbar_13TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['TTbarLepton_13']=gen2015('TTbarLepton_13TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['ZEE_13']=gen2015('ZEE_13TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['Wjet_Pt_80_120_13']=gen2015('Wjet_Pt_80_120_13TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['Wjet_Pt_3000_3500_13']=gen2015('Wjet_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['LM1_sfts_13']=gen2015('LM1_sfts_13TeV_cfi',Kby(9,100)) # this is to be removed once the next line is tested.
steps['SMS-T1tttt_mGl-1500_mLSP-100_13']=gen2015('SMS-T1tttt_mGl-1500_mLSP-100_13TeV-pythia8_cfi',Kby(9,50))
steps['QCD_FlatPt_15_3000_13']=gen2015('QCDForPF_13TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['QCD_FlatPt_15_3000HS_13']=gen2015('QCDForPF_13TeV_TuneCUETP8M1_cfi',Kby(50,100))

steps['ZpMM_2250_8TeV']=gen('ZpMM_2250_8TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['ZpEE_2250_8TeV']=gen('ZpEE_2250_8TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['ZpTT_1500_8TeV']=gen('ZpTT_1500_8TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['ZpMM_2250_13TeV']=gen2015('ZpMM_2250_13TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['ZpEE_2250_13TeV']=gen2015('ZpEE_2250_13TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['ZpTT_1500_13TeV']=gen2015('ZpTT_1500_13TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['HSCPstop_M_200_13TeV']=gen2015('HSCPstop_M_200_TuneCUETP8M1_13TeV_pythia8_cff',Kby(9,100))
steps['RSGravitonToGaGa_13TeV']=gen2015('RSGravitonToGammaGamma_kMpl01_M_3000_TuneCUETP8M1_13TeV_pythia8_cfi',Kby(9,100))
steps['WpToENu_M-2000_13TeV']=gen2015('WprimeToENu_M-2000_TuneCUETP8M1_13TeV-pythia8_cff',Kby(9,100))

def identitySim(wf):
    return merge([{'--restoreRND':'SIM','--process':'SIM2', '--inputCommands':'"keep *","drop *TagInfo*_*_*_*"' },wf])

steps['SingleMuPt10_UP15_ID']=identitySim(steps['SingleMuPt10_UP15'])
steps['TTbar_13_ID']=identitySim(steps['TTbar_13'])

baseDataSetRelease=[
    'CMSSW_7_1_0_pre7-PRE_STA71_V3-v1',                     # 0 run1 samples; keep GEN-SIM fixed to 710_pre7, for samples not routinely produced
    'CMSSW_7_4_0_pre7-MCHI2_74_V3-v1',                      # 1 Run1 HI GEN-SIM (only MB = wf 140)
    'CMSSW_6_2_0_pre8-PRE_ST62_V8_FastSim-v1',              # 2 for fastsim id test
#    'CMSSW_7_1_0_pre5-START71_V1-v2',                      # 3 8 TeV , for the one sample which is part of the routine relval production (RelValZmumuJets_Pt_20_300, because of -v2)
                                                            # THIS ABOVE IS NOT USED, AT THE MOMENT
    'CMSSW_7_4_0_pre7-MCRUN2_74_V7-v1',                     # 3 - 13 TeV samples with GEN-SIM from 740_p6; also GEN-SIM-DIGI-RAW-HLTDEBUG for id tests
    'CMSSW_7_3_0_pre1-PRE_LS172_V15_FastSim-v1',            # 4 - fast sim GEN-SIM-DIGI-RAW-HLTDEBUG for id tests
    'CMSSW_7_4_6_patch1-PU25ns_MCRUN2_74_V9-v1',                   # 5 - fullSim premix 25ns
    'CMSSW_7_4_6_patch1-PU50ns_MCRUN2_74_V8-v1',                   # 6 - fullSim premix 50ns
    'CMSSW_7_4_6_patch1-MCRUN2_74_V9_FastSim-v1'                   # 7 - fastSim minibias for mixing
    ]

# note: INPUT commands to be added once GEN-SIM w/ 13TeV+PostLS1Geo will be available 
steps['MinBiasINPUT']={'INPUT':InputInfo(dataSet='/RelValMinBias/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')} #was [0] 
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
steps['SingleMuPt10_UP15IDINPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt10_UP15/%s/GEN-SIM-DIGI-RAW-HLTDEBUG'%(baseDataSetRelease[3],),location='STD',split=1)}
steps['SingleMuPt10_UP15FSIDINPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt10/%s/GEN-SIM-DIGI-RECO'%(baseDataSetRelease[4],),location='STD',split=1)}
steps['SingleMuPt100INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt100/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['SingleMuPt1000INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt1000/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['TTbarINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['TTbar_13IDINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar_13/%s/GEN-SIM-DIGI-RAW-HLTDEBUG'%(baseDataSetRelease[3],),location='STD',split=1)}
steps['TTbar_13FSIDINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar_13/%s/GEN-SIM-DIGI-RECO'%(baseDataSetRelease[4],),location='STD',split=1)}
steps['TTbarLeptonINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbarLepton/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['OldTTbarINPUT']={'INPUT':InputInfo(dataSet='/RelValProdTTbar/CMSSW_5_0_0_pre6-START50_V5-v1/GEN-SIM-RECO',location='STD')}
steps['OldGenSimINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/CMSSW_4_4_2-START44_V7-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',location='STD')}
steps['Wjet_Pt_80_120INPUT']={'INPUT':InputInfo(dataSet='/RelValWjet_Pt_80_120/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['Wjet_Pt_3000_3500INPUT']={'INPUT':InputInfo(dataSet='/RelValWjet_Pt_3000_3500/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['LM1_sftsINPUT']={'INPUT':InputInfo(dataSet='/RelValLM1_sfts/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['QCD_FlatPt_15_3000INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_FlatPt_15_3000/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}

steps['QCD_FlatPt_15_3000HSINPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_FlatPt_15_3000HS/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['TTbar__DIGIPU1INPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/CMSSW_5_2_2-PU_START52_V4_special_120326-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',location='STD')}

# 13 TeV recycle GEN-SIM input
steps['MinBias_13INPUT']={'INPUT':InputInfo(dataSet='/RelValMinBias_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['QCD_Pt_3000_3500_13INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_3000_3500_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['QCD_Pt_600_800_13INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_600_800_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['QCD_Pt_80_120_13INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_80_120_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['QCD_Pt_80_120_13_HIINPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_80_120_13_HI/%s/GEN-SIM'%(baseDataSetRelease[1],),location='STD')}
steps['TTbar_13INPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['TTbarLepton_13INPUT']={'INPUT':InputInfo(dataSet='/RelValTTbarLepton_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['ZEE_13INPUT']={'INPUT':InputInfo(dataSet='/RelValZEE_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['Wjet_Pt_80_120_13INPUT']={'INPUT':InputInfo(dataSet='/RelValWjet_Pt_80_120_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['Wjet_Pt_3000_3500_13INPUT']={'INPUT':InputInfo(dataSet='/RelValWjet_Pt_3000_3500_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['LM1_sfts_13INPUT']={'INPUT':InputInfo(dataSet='/RelValLM1_sfts_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}#to be removed once the following line is tested. 
steps['SMS-T1tttt_mGl-1500_mLSP-100_13INPUT']={'INPUT':InputInfo(dataSet='/RelValSMS-T1tttt_mGl-1500_mLSP-100_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['QCD_FlatPt_15_3000_13INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_FlatPt_15_3000_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['QCD_FlatPt_15_3000HS_13INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_FlatPt_15_3000HS_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['ZpMM_2250_13TeVINPUT']={'INPUT':InputInfo(dataSet='/RelValZpMM_2250_13TeV/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['ZpEE_2250_13TeVINPUT']={'INPUT':InputInfo(dataSet='/RelValZpEE_2250_13TeV/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['ZpTT_1500_13TeVINPUT']={'INPUT':InputInfo(dataSet='/RelValZpTT_1500_13TeV/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['MinBiasHS_13INPUT']={'INPUT':InputInfo(dataSet='/RelValMinBiasHS_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['Higgs200ChargedTaus_13INPUT']={'INPUT':InputInfo(dataSet='/RelValHiggs200ChargedTaus_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}

# activate GEN-SIM recycling once we'll have the first set of gen-sim
steps['Upsilon1SToMuMu_13INPUT']={'INPUT':InputInfo(dataSet='/RelValUpsilon1SToMuMu_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['BuToKstarMuMu_13INPUT']={'INPUT':InputInfo(dataSet='/RelValBuToKstarMuMu_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['BsToMuMu_13INPUT']={'INPUT':InputInfo(dataSet='/RelValBsToMuMu_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['JpsiMuMu_Pt-15INPUT']={'INPUT':InputInfo(dataSet='/RelValJpsiMuMu_Pt-15/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['BuToKstarPsi2S_13INPUT']={'INPUT':InputInfo(dataSet='/RelValBuToKstarPsi2S_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['WE_13INPUT']={'INPUT':InputInfo(dataSet='/RelValWE_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['WM_13INPUT']={'INPUT':InputInfo(dataSet='/RelValWM_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['WpM_13INPUT']={'INPUT':InputInfo(dataSet='/RelValWpM_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['ZMM_13INPUT']={'INPUT':InputInfo(dataSet='/RelValZMM_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['ZMM_13_HIINPUT']={'INPUT':InputInfo(dataSet='/RelValZMM_13_HI/%s/GEN-SIM'%(baseDataSetRelease[1],),location='STD')}
steps['ZEEMM_13_HIINPUT']={'INPUT':InputInfo(dataSet='/RelValZEEMM_13_HI/%s/GEN-SIM'%(baseDataSetRelease[1],),location='STD')}
steps['ZpMM_13INPUT']={'INPUT':InputInfo(dataSet='/RelValZpMM_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['ZTT_13INPUT']={'INPUT':InputInfo(dataSet='/RelValZTT_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['H130GGgluonfusion_13INPUT']={'INPUT':InputInfo(dataSet='/RelValH130GGgluonfusion_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['PhotonJets_Pt_10_13INPUT']={'INPUT':InputInfo(dataSet='/RelValPhotonJets_Pt_10_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['PhotonJets_Pt_10_13_HIINPUT']={'INPUT':InputInfo(dataSet='/RelValPhotonJets_Pt_10_13_HI/%s/GEN-SIM'%(baseDataSetRelease[1],),location='STD')}
steps['QQH1352T_13INPUT']={'INPUT':InputInfo(dataSet='/RelValQQH1352T_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['ADDMonoJet_d3MD3_13INPUT']={'INPUT':InputInfo(dataSet='/RelValADDMonoJet_d3MD3_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['RSKKGluon_m3000GeV_13INPUT']={'INPUT':InputInfo(dataSet='/RelValRSKKGluon_m3000GeV_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['BuJpsiK_13INPUT']={'INPUT':InputInfo(dataSet='/RelValBuJpsiK_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['Cosmics_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValCosmics_UP15/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['BeamHalo_13INPUT']={'INPUT':InputInfo(dataSet='/RelValBeamHalo_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['HSCPstop_M_200_13TeVINPUT']={'INPUT':InputInfo(dataSet='/RelValHSCPstop_M_200_13TeV/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['RSGravitonToGaGa_13TeVINPUT']={'INPUT':InputInfo(dataSet='/RelValRSGravitonToGaGa_13TeV/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['WpToENu_M-2000_13TeVINPUT']={'INPUT':InputInfo(dataSet='/RelValWpToENu_M-2000_13TeV/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}

# particle guns with postLS1 geometry recycle GEN-SIM input
steps['SingleElectronPt10_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleElectronPt10_UP15/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['SingleElectronPt35_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleElectronPt35_UP15/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['SingleElectronPt1000_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleElectronPt1000_UP15/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['SingleElectronFlatPt1To100_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleElectronFlatPt1To100_UP15/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['SingleGammaPt10_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleGammaPt10_UP15/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['SingleGammaPt35_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleGammaPt35_UP15/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['SingleMuPt1_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt1_UP15/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['SingleMuPt10_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt10_UP15/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['SingleMuPt100_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt100_UP15/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['SingleMuPt1000_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt1000_UP15/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['NuGun_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValNuGun_UP15/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}

#input for fast sim workflows to be added - TODO


## high stat step1
ecalHcal={
    '-s':'GEN,SIM,DIGI,DIGI2RAW,RAW2DIGI,L1Reco,RECO,EI',
    '--datatier':'GEN-SIM-DIGI-RAW-RECO',
    #'--geometry':'ECALHCAL',
    '--eventcontent':'FEVTDEBUG',
    '--customise':'Validation/Configuration/ECALHCAL.customise,SimGeneral/MixingModule/fullMixCustomize_cff.setCrossingFrameOn',
    '--beamspot':'NoSmear'}

steps['SingleElectronE120EHCAL']=merge([{'cfg':'SingleElectronE120EHCAL_pythia8_cfi'},ecalHcal,Kby(25,250),step1Defaults])
steps['SinglePiE50HCAL']=merge([{'cfg':'SinglePiE50HCAL_pythia8_cfi'},ecalHcal,Kby(25,250),step1Defaults])

steps['MinBiasHS']=gen('MinBias_8TeV_pythia8_TuneCUETP8M1_cff',Kby(25,300))
steps['InclusiveppMuX']=gen('InclusiveppMuX_8TeV_TuneCUETP8M1_cfi',Mby(11,45000))
steps['SingleElectronFlatPt5To100']=gen('SingleElectronFlatPt5To100_pythia8_cfi',Kby(25,250))
steps['SinglePiPt1']=gen('SinglePiPt1_pythia8_cfi',Kby(25,250))
steps['SingleMuPt1HS']=gen('SingleMuPt1_pythia8_cfi',Kby(25,1000))
steps['ZPrime5000Dijet']=gen('ZPrime5000JJ_8TeV_TuneCUETP8M1_cfi',Kby(25,100))
steps['SinglePi0E10']=gen('SinglePi0E10_pythia8_cfi',Kby(25,100))
steps['SinglePiPt10']=gen('SinglePiPt10_pythia8_cfi',Kby(25,250))
steps['SingleGammaFlatPt10To100']=gen('SingleGammaFlatPt10To100_pythia8_cfi',Kby(25,250))
steps['SingleTauPt50Pythia']=gen('SingleTaupt_50_pythia8_cfi',Kby(25,100))
steps['SinglePiPt100']=gen('SinglePiPt100_pythia8_cfi',Kby(25,250))


def genS(fragment,howMuch):
    global step1Defaults,stCond
    return merge([{'cfg':fragment},stCond,howMuch,step1Defaults])

steps['Higgs200ChargedTaus']=genS('H200ChargedTaus_Tauola_8TeV_cfi',Kby(9,100))
steps['JpsiMM']=genS('JpsiMM_8TeV_TuneCUETP8M1_cfi',Kby(66,1000))
steps['WE']=genS('WE_8TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['WM']=genS('WM_8TeV_TuneCUETP8M1_cfi',Kby(9,200))
steps['WpM']=genS('WpM_8TeV_TuneCUETP8M1_cfi',Kby(9,200))
steps['ZMM']=genS('ZMM_8TeV_TuneCUETP8M1_cfi',Kby(18,300))
steps['ZpMM']=genS('ZpMM_8TeV_TuneCUETP8M1_cfi',Kby(9,200))
steps['Higgs200ChargedTaus_13']=gen2015('H200ChargedTaus_Tauola_13TeV_cfi',Kby(9,100))
steps['Upsilon1SToMuMu_13']=gen2015('Upsilon1SToMuMu_forSTEAM_13TeV_TuneCUETP8M1_cfi',Kby(17,190)) 
steps['BuToKstarMuMu_13']=gen2015('BuToKstarMuMu_forSTEAM_13TeV_TuneCUETP8M1_cfi',Kby(2250,25000))
steps['BsToMuMu_13']=gen2015('BsToMuMu_forSTEAM_13TeV_TuneCUETP8M1_cfi',Kby(30000,333333))
steps['JpsiMuMu_Pt-15']=gen2015('JpsiMuMu_Pt-15_forSTEAM_13TeV_cfi',Kby(11000,122000))
steps['BuToKstarPsi2S_13']=gen2015('BuToKstarPsi2S_forSTEAM_13TeV_TuneCUETP8M1_cfi',Kby(16000,176000))
steps['WE_13']=gen2015('WE_13TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['WM_13']=gen2015('WM_13TeV_TuneCUETP8M1_cfi',Kby(9,200))
steps['WpM_13']=gen2015('WpM_13TeV_TuneCUETP8M1_cfi',Kby(9,200))
steps['ZMM_13']=gen2015('ZMM_13TeV_TuneCUETP8M1_cfi',Kby(18,300))
steps['ZEEMM_13']=gen2015('ZEEMM_13TeV_TuneCUETP8M1_cfi',Kby(18,300))
steps['ZpMM_13']=gen2015('ZpMM_13TeV_TuneCUETP8M1_cfi',Kby(9,200))

steps['ZTT']=genS('ZTT_All_hadronic_8TeV_TuneCUETP8M1_cfi',Kby(9,150))
steps['H130GGgluonfusion']=genS('H130GGgluonfusion_8TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['PhotonJets_Pt_10']=genS('PhotonJet_Pt_10_8TeV_TuneCUETP8M1_cfi',Kby(9,150))
steps['QQH1352T']=genS('QQH1352T_8TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['ZTT_13']=gen2015('ZTT_All_hadronic_13TeV_TuneCUETP8M1_cfi',Kby(9,150))
steps['H130GGgluonfusion_13']=gen2015('H130GGgluonfusion_13TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['PhotonJets_Pt_10_13']=gen2015('PhotonJet_Pt_10_13TeV_TuneCUETP8M1_cfi',Kby(9,150))
steps['QQH1352T_13']=gen2015('QQH1352T_13TeV_TuneCUETP8M1_cfi',Kby(9,100))
#steps['ZmumuJets_Pt_20_300']=gen('ZmumuJets_Pt_20_300_GEN_8TeV_TuneCUETP8M1_cfg',Kby(25,100))
steps['ADDMonoJet_d3MD3']=genS('ADDMonoJet_8TeV_d3MD3_TuneCUETP8M1_cfi',Kby(9,100))
steps['ADDMonoJet_d3MD3_13']=gen2015('ADDMonoJet_13TeV_d3MD3_TuneCUETP8M1_cfi',Kby(9,100))
steps['RSKKGluon_m3000GeV_13']=gen2015('RSKKGluon_m3000GeV_13TeV_TuneCUETP8M1_cff',Kby(9,100))
steps['BuJpsiK_13']=gen2015('Pythia8_BuJpsiK_13TeV_TuneCUETP8M1_cfi',Kby(36000,400000))

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
steps['QQH1352TINPUT']={'INPUT':InputInfo(dataSet='/RelValQQH1352T_Tauola/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['ADDMonoJet_d3MD3INPUT']={'INPUT':InputInfo(dataSet='/RelValADDMonoJet_d3MD3/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['WpMINPUT']={'INPUT':InputInfo(dataSet='/RelValWpM/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['ZpMMINPUT']={'INPUT':InputInfo(dataSet='/RelValZpMM/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['ZpMM_2250_8TeVINPUT']={'INPUT':InputInfo(dataSet='/RelValZpMM_2250_8TeV_Tauola/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['ZpEE_2250_8TeVINPUT']={'INPUT':InputInfo(dataSet='/RelValZpEE_2250_8TeV_Tauola/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['ZpTT_1500_8TeVINPUT']={'INPUT':InputInfo(dataSet='/RelValZpTT_1500_8TeV_Tauola/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}


steps['Cosmics']=merge([{'cfg':'UndergroundCosmicMu_cfi.py','--scenario':'cosmics'},Kby(666,100000),step1Defaults])
steps['Cosmics_UP15']=merge([{'cfg':'UndergroundCosmicMu_cfi.py','--scenario':'cosmics'},Kby(666,100000),step1Up2015Defaults])
steps['BeamHalo']=merge([{'cfg':'BeamHalo_cfi.py','--scenario':'cosmics'},Kby(9,100),step1Defaults])
steps['BeamHalo_13']=merge([{'cfg':'BeamHalo_13TeV_cfi.py','--scenario':'cosmics'},Kby(9,100),step1Up2015Defaults])

# GF re-introduce INPUT command once GEN-SIM will be available
# steps['CosmicsINPUT']={'INPUT':InputInfo(dataSet='/RelValCosmics/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['BeamHaloINPUT']={'INPUT':InputInfo(dataSet='/RelValBeamHalo/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}

steps['QCD_Pt_50_80']=genS('QCD_Pt_50_80_8TeV_TuneCUETP8M1_cfi',Kby(25,100))
steps['QCD_Pt_15_20']=genS('QCD_Pt_15_20_8TeV_TuneCUETP8M1_cfi',Kby(25,100))
steps['ZTTHS']=merge([Kby(25,100),steps['ZTT']])
steps['QQH120Inv']=genS('QQH120Inv_8TeV_TuneCUETP8M1_cfi',Kby(25,100))
steps['TTbar2HS']=merge([Kby(25,100),steps['TTbar']])
steps['JpsiMM_Pt_20_inf']=genS('JpsiMM_Pt_20_inf_8TeV_TuneCUETP8M1_cfi',Kby(70,280))
steps['QCD_Pt_120_170']=genS('QCD_Pt_120_170_8TeV_TuneCUETP8M1_cfi',Kby(25,100))
steps['H165WW2L']=genS('H165WW2L_8TeV_TuneCUETP8M1_cfi',Kby(25,100))
steps['UpsMM']=genS('UpsMM_8TeV_TuneCUETP8M1_cfi',Kby(56250,225))
steps['RSGrav']=genS('RS750_quarks_and_leptons_8TeV_TuneCUETP8M1_cff',Kby(25,100))
steps['QCD_Pt_80_120_2HS']=merge([Kby(25,100),steps['QCD_Pt_80_120']])
steps['bJpsiX']=genS('bJpsiX_8TeV_TuneCUETP8M1_cfi',Mby(325,1300000))
steps['QCD_Pt_30_50']=genS('QCD_Pt_30_50_8TeV_TuneCUETP8M1_cfi',Kby(25,100))
steps['H200ZZ4L']=genS('H200ZZ4L_8TeV_TuneCUETP8M1_cfi',Kby(25,100))
steps['LM9p']=genS('LM9p_8TeV_cff',Kby(25,100))
steps['QCD_Pt_20_30']=genS('QCD_Pt_20_30_8TeV_TuneCUETP8M1_cfi',Kby(25,100))
steps['QCD_Pt_170_230']=genS('QCD_Pt_170_230_8TeV_TuneCUETP8M1_cfi',Kby(25,100))




## pPb tests
step1PPbDefaults={'--beamspot':'Realistic8TeVCollision'}
steps['AMPT_PPb_5020GeV_MinimumBias']=merge([{'-n':10},step1PPbDefaults,genS('AMPT_PPb_5020GeV_MinimumBias_cfi',Kby(9,100))])
# GF to be uncommented when GEN-SIM becomes available
# steps['AMPT_PPb_5020GeV_MinimumBiasINPUT']={'INPUT':InputInfo(dataSet='/RelValAMPT_PPb_5020GeV_MinimumBias/%s/GEN-SIM'%(baseDataSetRelease[4],),location='STD')}

## heavy ions tests
U2000by1={'--relval': '2000,1'}
U80by1={'--relval': '80,1'}

hiAlca = {'--conditions':'auto:run2_mc_HIon', '--customise':'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1_HI'}
hiDefaults=merge([hiAlca,{'--scenario':'HeavyIons','-n':2,'--beamspot':'RealisticHI2011Collision'}])

steps['HydjetQ_MinBias_2760GeV']=merge([{'-n':1},hiDefaults,genS('Hydjet_Quenched_MinBias_2760GeV_cfi',U2000by1)])
steps['HydjetQ_MinBias_2760GeVINPUT']={'INPUT':InputInfo(dataSet='/RelValHydjetQ_MinBias_2760GeV/%s/GEN-SIM'%(baseDataSetRelease[1],),location='STD',split=5)}
steps['HydjetQ_MinBias_2760GeV_UP15']=merge([{'-n':1},hiDefaults,genS('Hydjet_Quenched_MinBias_2760GeV_cfi',U2000by1)])
steps['HydjetQ_MinBias_2760GeV_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValHydjetQ_MinBias_2760GeV/%s/GEN-SIM'%(baseDataSetRelease[1],),location='STD',split=5)}
#steps['HydjetQ_B8_2760GeV']=merge([{'-n':1},hiDefaults,genS('Hydjet_Quenched_B8_2760GeV_cfi',U80by1)])
#steps['HydjetQ_B8_2760GeVINPUT']={'INPUT':InputInfo(dataSet='/RelValHydjetQ_B8_2760GeV/%s/GEN-SIM'%(baseDataSetRelease[x],),location='CAF')}
steps['QCD_Pt_80_120_13_HI']=merge([hiDefaults,steps['QCD_Pt_80_120_13']])
steps['PhotonJets_Pt_10_13_HI']=merge([hiDefaults,steps['PhotonJets_Pt_10_13']])
steps['ZMM_13_HI']=merge([hiDefaults,steps['ZMM_13']])
steps['ZEEMM_13_HI']=merge([hiDefaults,steps['ZEEMM_13']])


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
step1FastDefaults =merge([{'-s':'GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,L1,L1Reco,RECO,EI,HLT:@fake,VALIDATION',
                           '--fast':'',
                           '--beamspot'    : 'Realistic8TeVCollision',
                           '--eventcontent':'FEVTDEBUGHLT,DQM',
                           '--datatier':'GEN-SIM-DIGI-RECO,DQMIO',
                           '--relval':'27000,3000'},
                          step1Defaults])
step1FastUpg2015Defaults =merge([{'-s':'GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,L1,L1Reco,RECO,EI,HLT:@frozen25ns,VALIDATION',
                           '--fast':'',
                           '--conditions'  :'auto:run2_mc',
                           '--beamspot'    : 'NominalCollision2015',
                           '--magField'    :'38T_PostLS1',
                           '--customise'   :'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1',
                           '--eventcontent':'FEVTDEBUGHLT,DQM',
                           '--datatier':'GEN-SIM-DIGI-RECO,DQMIO',
                           '--relval':'27000,3000'},
                           step1Defaults])
step1FastPUNewMixing =merge([{'-s':'GEN,SIM,RECOBEFMIX',
                           '--eventcontent':'FASTPU',
                           '--datatier':'GEN-SIM-RECO'},
                           step1FastUpg2015Defaults])


#step1FastDefaults
steps['TTbarFS']=merge([{'cfg':'TTbar_8TeV_TuneCUETP8M1_cfi'},Kby(100,1000),step1FastDefaults])
steps['SingleMuPt1FS']=merge([{'cfg':'SingleMuPt1_pythia8_cfi'},step1FastDefaults])
steps['SingleMuPt10FS']=merge([{'cfg':'SingleMuPt10_pythia8_cfi'},step1FastDefaults])
steps['SingleMuPt100FS']=merge([{'cfg':'SingleMuPt100_pythia8_cfi'},step1FastDefaults])
steps['SinglePiPt1FS']=merge([{'cfg':'SinglePiPt1_pythia8_cfi'},step1FastDefaults])
steps['SinglePiPt10FS']=merge([{'cfg':'SinglePiPt10_pythia8_cfi'},step1FastDefaults])
steps['SinglePiPt100FS']=merge([{'cfg':'SinglePiPt100_pythia8_cfi'},step1FastDefaults])
steps['ZEEFS']=merge([{'cfg':'ZEE_8TeV_TuneCUETP8M1_cfi'},Kby(100,2000),step1FastDefaults])
steps['ZTTFS']=merge([{'cfg':'ZTT_Tauola_OneLepton_OtherHadrons_8TeV_TuneCUETP8M1_cfi'},Kby(100,2000),step1FastDefaults])
steps['QCDFlatPt153000FS']=merge([{'cfg':'QCDForPF_8TeV_TuneCUETP8M1_cfi'},Kby(27,2000),step1FastDefaults])
steps['QCD_Pt_80_120FS']=merge([{'cfg':'QCD_Pt_80_120_8TeV_TuneCUETP8M1_cfi'},Kby(100,500),stCond,step1FastDefaults])
steps['QCD_Pt_3000_3500FS']=merge([{'cfg':'QCD_Pt_3000_3500_8TeV_TuneCUETP8M1_cfi'},Kby(100,500),stCond,step1FastDefaults])
steps['H130GGgluonfusionFS']=merge([{'cfg':'H130GGgluonfusion_8TeV_TuneCUETP8M1_cfi'},step1FastDefaults])
steps['SingleGammaFlatPt10To10FS']=merge([{'cfg':'SingleGammaFlatPt10To100_pythia8_cfi'},Kby(100,500),step1FastDefaults])

#step1FastUpg2015Defaults
steps['TTbarFS_13']=merge([{'cfg':'TTbar_13TeV_TuneCUETP8M1_cfi'},Kby(100,1000),step1FastUpg2015Defaults])
steps['SMS-T1tttt_mGl-1500_mLSP-100FS_13']=merge([{'cfg':'SMS-T1tttt_mGl-1500_mLSP-100_13TeV-pythia8_cfi'},Kby(100,1000),step1FastUpg2015Defaults])
steps['NuGunFS_UP15']=merge([{'cfg':'SingleNuE10_cfi'},Kby(100,1000),step1FastUpg2015Defaults])
steps['ZEEFS_13']=merge([{'cfg':'ZEE_13TeV_TuneCUETP8M1_cfi'},Kby(100,2000),step1FastUpg2015Defaults])
steps['ZTTFS_13']=merge([{'cfg':'ZTT_All_hadronic_13TeV_TuneCUETP8M1_cfi'},Kby(100,2000),step1FastUpg2015Defaults])
steps['ZMMFS_13']=merge([{'cfg':'ZMM_13TeV_TuneCUETP8M1_cfi'},Kby(100,2000),step1FastUpg2015Defaults])
steps['QCDFlatPt153000FS_13']=merge([{'cfg':'QCDForPF_13TeV_TuneCUETP8M1_cfi'},Kby(27,2000),step1FastUpg2015Defaults])
steps['QCD_Pt_80_120FS_13']=merge([{'cfg':'QCD_Pt_80_120_13TeV_TuneCUETP8M1_cfi'},Kby(100,500),step1FastUpg2015Defaults])
steps['QCD_Pt_3000_3500FS_13']=merge([{'cfg':'QCD_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi'},Kby(100,500),step1FastUpg2015Defaults])
steps['H130GGgluonfusionFS_13']=merge([{'cfg':'H130GGgluonfusion_13TeV_TuneCUETP8M1_cfi'},step1FastUpg2015Defaults])
steps['SingleMuPt10FS_UP15']=merge([{'cfg':'SingleMuPt10_pythia8_cfi'},step1FastUpg2015Defaults])
steps['SingleMuPt100FS_UP15']=merge([{'cfg':'SingleMuPt100_pythia8_cfi'},step1FastUpg2015Defaults])

#step1FastPU
steps['MinBiasFS_13_ForMixing']=merge([{'cfg':'MinBias_13TeV_pythia8_TuneCUETP8M1_cfi'},Kby(100,1000),step1FastPUNewMixing])

steps['TTbarSFS']=merge([{'cfg':'TTbar_8TeV_TuneCUETP8M1_cfi'},
                        {'-s':'GEN,SIM',
                         '--eventcontent':'FEVTDEBUG',
                         '--datatier':'GEN-SIM',
                         '--fast':''},
                        step1Defaults])

steps['TTbarSFSA']=merge([{'cfg':'TTbar_8TeV_TuneCUETP8M1_cfi',
                           '-s':'GEN,SIM,RECO,EI,HLT:@fake,VALIDATION',
                           '--fast':''},
                          step1FastDefaults])

def identityFS(wf):
    return merge([{'--restoreRND':'HLT','--process':'HLT2','--hltProcess':'HLT2', '--inputCommands':'"keep *","drop *TagInfo*_*_*_*"'},wf])

steps['SingleMuPt10FS_UP15_ID']=identityFS(steps['SingleMuPt10FS_UP15'])
steps['TTbarFS_13_ID']=identityFS(steps['TTbarFS_13'])

step1GenDefaults=merge([{'-s':'GEN,VALIDATION:genvalid',
                         '--relval':'250000,20000',
                         '--eventcontent':'RAWSIM,DQM',
                         '--datatier':'GEN,DQMIO',
                         '--conditions':'auto:run2_mc_FULL'
                         },
                        step1Defaults])

step1HadronizerDefaults=merge([{'--datatier':'GEN-SIM,DQMIO'},step1GenDefaults])

step1LHEDefaults=merge([{'-s':'LHE',
                         '--relval':'200000,5000',
                         '--eventcontent':'LHE',
                         '--datatier':'GEN',
                         '--conditions':'auto:run2_mc_FULL'                         
                         },
                        step1Defaults])


def genvalid(fragment,d,suffix='all',fi='',dataSet=''):
    import copy
    c=copy.copy(d)
    if suffix:
        c['-s']=c['-s'].replace('genvalid','genvalid_'+suffix)
    if fi:
        c['--filein']='lhe:%d'%(fi,)
    if dataSet:
        c['--filein']='das:%s'%(dataSet,)
    c['cfg']=fragment
    return c



steps['DYToll01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV']=genvalid('Configuration/Generator/python/DYToll01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV_cff.py',step1LHEDefaults)
# all 6 workflows with root step 'DYToll01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV' will recycle the same dataset, from wf [512] of generator set
# steps['DYToll01234Jets_5f_LO_MLM_Madgraph_LHE_13TeVINPUT']={'INPUT':InputInfo(dataSet='/DYToll01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV_py8/CMSSW_7_4_0_pre0-MCRUN2_73_V5-v1/GEN',location='STD')}

steps['MinBias_TuneZ2star_13TeV_pythia6']=genvalid('MinBias_TuneZ2star_13TeV_pythia6_cff',step1GenDefaults)
steps['QCD_Pt-30_TuneZ2star_13TeV_pythia6']=genvalid('QCD_Pt_30_TuneZ2star_13TeV_pythia6_cff',step1GenDefaults)
steps['MinBias_13TeV_pythia8']=genvalid('MinBias_13TeV_pythia8_cff',step1GenDefaults)
steps['QCD_Pt-30_13TeV_pythia8']=genvalid('QCD_Pt_30_13TeV_pythia8_cff',step1GenDefaults)

steps['DYToLL_M-50_13TeV_pythia8']=genvalid('DYToLL_M-50_13TeV_pythia8_cff',step1GenDefaults)
steps['WToLNu_13TeV_pythia8']=genvalid('WToLNu_13TeV_pythia8_cff',step1GenDefaults)

steps['SoftQCDDiffractive_13TeV_pythia8']=genvalid('SoftQCDDiffractive_13TeV_pythia8_cff',step1GenDefaults)
steps['SoftQCDnonDiffractive_13TeV_pythia8']=genvalid('SoftQCDnonDiffractive_13TeV_pythia8_cff',step1GenDefaults)
steps['SoftQCDelastic_13TeV_pythia8']=genvalid('SoftQCDelastic_13TeV_pythia8_cff',step1GenDefaults)
steps['SoftQCDinelastic_13TeV_pythia8']=genvalid('SoftQCDinelastic_13TeV_pythia8_cff',step1GenDefaults)

steps['QCD_Pt-30_8TeV_herwigpp']=genvalid('QCD_Pt_30_8TeV_herwigpp_cff',step1GenDefaults)

# Generator Hadronization (Hadronization of LHE)
steps['WJetsLNu_13TeV_madgraph-pythia8']=genvalid('Hadronizer_MgmMatchTuneCUETP8M1_13TeV_madgraph_pythia8_cff',step1GenDefaults,dataSet='/WJetsToLNu_13TeV-madgraph/Fall13wmLHE-START62_V1-v1/GEN')
steps['Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_cff',step1HadronizerDefaults)
steps['GGToH_13TeV_pythia8']=genvalid('GGToHtautau_13TeV_pythia8_cff',step1GenDefaults)

steps['WJetsLNutaupinu_13TeV_madgraph-pythia8']=genvalid('Hadronizer_MgmMatchTuneCUETP8M1_13TeV_madgraph_pythia8_taupinu_cff',step1GenDefaults,dataSet='/WJetsToLNu_13TeV-madgraph/Fall13wmLHE-START62_V1-v1/GEN')
steps['Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_taupinu']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_taupinu_cff',step1HadronizerDefaults)
steps['GGToHtaupinu_13TeV_pythia8']=genvalid('GGToHtautau_13TeV_pythia8_taupinu_cff',step1GenDefaults)

steps['WJetsLNutaurhonu_13TeV_madgraph-pythia8']=genvalid('Hadronizer_MgmMatchTuneCUETP8M1_13TeV_madgraph_pythia8_taurhonu_cff.py',step1GenDefaults,dataSet='/WJetsToLNu_13TeV-madgraph/Fall13wmLHE-START62_V1-v1/GEN')
steps['Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_taurhonu']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_taurhonu_cff.py',step1HadronizerDefaults)
steps['GGToHtaurhonu_13TeV_pythia8']=genvalid('GGToHtautau_13TeV_pythia8_taurhonu_cff',step1GenDefaults)

# Generator External Decays
steps['TT_13TeV_pythia8-evtgen']=genvalid('Hadronizer_MgmMatchTuneCUETP8M1_13TeV_madgraph_pythia8_EvtGen_cff',step1GenDefaults,dataSet='/TTJets_MSDecaysCKM_central_13TeV-madgraph/Fall13wmLHE-START62_V1-v1/GEN')

steps['Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola_cff',step1HadronizerDefaults)
steps['WToLNu_13TeV_pythia8-tauola']=genvalid('Hadronizer_MgmMatchTuneCUETP8M1_13TeV_madgraph_pythia8_Tauola_cff',step1GenDefaults,dataSet='/WJetsToLNu_13TeV-madgraph/Fall13wmLHE-START62_V1-v1/GEN')
steps['GGToH_13TeV_pythia8-tauola']=genvalid('GGToHtautau_13TeV_pythia8_Tauola_cff',step1GenDefaults)

steps['WToLNutaupinu_13TeV_pythia8-tauola']=genvalid('Hadronizer_MgmMatchTuneCUETP8M1_13TeV_madgraph_pythia8_Tauola_taupinu_cff',step1GenDefaults,dataSet='/WJetsToLNu_13TeV-madgraph/Fall13wmLHE-START62_V1-v1/GEN')
steps['Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola_taupinu']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola_taupinu_cff',step1HadronizerDefaults)
steps['GGToHtaupinu_13TeV_pythia8-tauola']=genvalid('GGToHtautau_13TeV_pythia8_Tauola_taupinu_cff',step1GenDefaults)

steps['WToLNutaurhonu_13TeV_pythia8-tauola']=genvalid('Hadronizer_MgmMatchTuneCUETP8M1_13TeV_madgraph_pythia8_Tauola_taurhonu_cff',step1GenDefaults,dataSet='/WJetsToLNu_13TeV-madgraph/Fall13wmLHE-START62_V1-v1/GEN')
steps['Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola_taurhonu']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola_taurhonu_cff',step1HadronizerDefaults)
steps['GGToHtaurhonu_13TeV_pythia8-tauola']=genvalid('GGToHtautau_13TeV_pythia8_Tauola_taurhonu_cff',step1GenDefaults)

# Heavy Ion
steps['ReggeGribovPartonMC_EposLHC_5TeV_pPb']=genvalid('GeneratorInterface/ReggeGribovPartonMCInterface/ReggeGribovPartonMC_EposLHC_5TeV_pPb_cfi',step1GenDefaults)




#PU for FullSim
PU={'-n':10,'--pileup':'default','--pileup_input':'das:/RelValMinBias/%s/GEN-SIM'%(baseDataSetRelease[0],)}
# pu2 can be removed
PU2={'-n':10,'--pileup':'default','--pileup_input':'das:/RelValMinBias/%s/GEN-SIM'%(baseDataSetRelease[0],)}
PU25={'-n':10,'--pileup':'AVE_35_BX_25ns','--pileup_input':'das:/RelValMinBias_13/%s/GEN-SIM'%(baseDataSetRelease[3],)}
PU50={'-n':10,'--pileup':'AVE_35_BX_50ns','--pileup_input':'das:/RelValMinBias_13/%s/GEN-SIM'%(baseDataSetRelease[3],)}



#PU for FastSim
# FS_PU_INPUT_13TEV = "file:/afs/cern.ch/work/l/lveldere/minbias.root" # placeholder for relval to be produced with wf  135.8
PUFS={'--pileup':'GEN_2012_Summer_50ns_PoissonOOTPU'}
# PUFS2={'--pileup':'2012_Startup_50ns_PoissonOOTPU'} # not used anywhere
PUFSAVE10={'--pileup':'GEN_AVE_10_BX_25ns'}  # temporary: one or a few releases as back-up
PUFSAVE20={'--pileup':'GEN_AVE_20_BX_25ns'}  # temporary: one or a few releases as back-up
PUFSAVE35={'--pileup':'GEN_AVE_35_BX_25ns'}
PUFSAVE10_DRMIX_ITO={'--pileup':'AVE_10_BX_25ns','--pileup_input':'das:/RelValMinBiasFS_13_ForMixing/%s/GEN-SIM-RECO'%(baseDataSetRelease[7],),'--customise':'FastSimulation/Configuration/Customs.disableOOTPU,SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1'}
PUFSAVE35_DRMIX_ITO={'--pileup':'AVE_35_BX_25ns','--pileup_input':'das:/RelValMinBiasFS_13_ForMixing/%s/GEN-SIM-RECO'%(baseDataSetRelease[7],),'--customise':'FastSimulation/Configuration/Customs.disableOOTPU,SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1'}
PUFS25={'--pileup':'AVE_35_BX_25ns','--pileup_input':'das:/RelValMinBiasFS_13_ForMixing/%s/GEN-SIM-RECO'%(baseDataSetRelease[7],)}


#
steps['TTbarFSPU']=merge([PUFS,Kby(100,500),steps['TTbarFS']] )

steps['FS_TTbar_13_PUAVE10']=merge([PUFSAVE10,Kby(100,500),steps['TTbarFS_13']] ) # temporary: one or a few releases as back-up
steps['FS_TTbar_13_PUAVE20']=merge([PUFSAVE20,Kby(100,500),steps['TTbarFS_13']] ) # temporary: one or a few releases as back-up
steps['FS_TTbar_13_PUAVE35']=merge([PUFSAVE35,Kby(100,500),steps['TTbarFS_13']] )
steps['FS_TTbar_13_PU25']=merge([PUFS25,Kby(100,500),steps['TTbarFS_13']] ) # needs the placeholder
steps['FS_NuGun_UP15_PU25']=merge([PUFS25,Kby(100,500),steps['NuGunFS_UP15']] ) # needs the placeholder
steps['FS_SMS-T1tttt_mGl-1500_mLSP-100_13_PU25']=merge([PUFS25,Kby(100,500),steps['SMS-T1tttt_mGl-1500_mLSP-100FS_13']] )
steps['FS_TTbar_13_PUAVE10_DRMIX_ITO']=merge([PUFSAVE10_DRMIX_ITO,Kby(100,500),steps['TTbarFS_13']] ) # needs the placeholder
steps['FS_TTbar_13_PUAVE35_DRMIX_ITO']=merge([PUFSAVE35_DRMIX_ITO,Kby(100,500),steps['TTbarFS_13']] ) # needs the placeholder

# step2 
step2Defaults = { '-s'            : 'DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@fake,RAW2DIGI,L1Reco',
                  '--datatier'    : 'GEN-SIM-DIGI-RAW-HLTDEBUG',
                  '--eventcontent': 'FEVTDEBUGHLT',
                  '--conditions'  : 'auto:run1_mc',
                  }
#for 2015
step2Upg2015Defaults = {'-s'     :'DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@frozen25ns,RAW2DIGI,L1Reco',
                 '--conditions'  :'auto:run2_mc',
                 '--magField'    :'38T_PostLS1',
                 '--datatier'    :'GEN-SIM-DIGI-RAW-HLTDEBUG',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--customise'   :'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1',
                 '-n'            :'10'
                  }
step2Upg2015Defaults50ns = merge([{'-s':'DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@frozen50ns,RAW2DIGI,L1Reco','--conditions':'auto:run2_mc_50ns','--customise':'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1_50ns'},step2Upg2015Defaults])

steps['DIGIUP15']=merge([step2Upg2015Defaults])
steps['DIGIUP15PROD1']=merge([{'-s':'DIGI,L1,DIGI2RAW,HLT:@frozen25ns,RAW2DIGI,L1Reco','--eventcontent':'RAWSIM','--datatier':'GEN-SIM-RAW'},step2Upg2015Defaults])
steps['DIGIUP15_PU25']=merge([PU25,step2Upg2015Defaults])
steps['DIGIUP15_PU50']=merge([PU50,step2Upg2015Defaults50ns])

steps['DIGIPROD1']=merge([{'-s':'DIGI,L1,DIGI2RAW,HLT:@fake,RAW2DIGI,L1Reco','--eventcontent':'RAWSIM','--datatier':'GEN-SIM-RAW'},step2Defaults])
steps['DIGI']=merge([step2Defaults])
#steps['DIGI2']=merge([stCond,step2Defaults])
steps['DIGICOS']=merge([{'--scenario':'cosmics','--eventcontent':'FEVTDEBUG','--datatier':'GEN-SIM-DIGI-RAW'},stCond,step2Defaults])
steps['DIGIHAL']=merge([{'--scenario':'cosmics','--eventcontent':'FEVTDEBUG','--datatier':'GEN-SIM-DIGI-RAW'},step2Upg2015Defaults])

steps['DIGIPU1']=merge([PU,step2Defaults])
steps['DIGIPU2']=merge([PU2,step2Defaults])
steps['REDIGIPU']=merge([{'-s':'reGEN,reDIGI,L1,DIGI2RAW,HLT:@fake,RAW2DIGI,L1Reco'},steps['DIGIPU1']])

steps['DIGIUP15_ID']=merge([{'--restoreRND':'HLT','--process':'HLT2'},steps['DIGIUP15']])

steps['RESIM']=merge([{'-s':'reGEN,reSIM','-n':10},steps['DIGI']])
#steps['RESIMDIGI']=merge([{'-s':'reGEN,reSIM,DIGI,L1,DIGI2RAW,HLT:@fake,RAW2DIGI,L1Reco','-n':10,'--restoreRNDSeeds':'','--process':'HLT'},steps['DIGI']])

    
steps['DIGIHI']=merge([{'--conditions':'auto:run2_mc_HIon', '-s':'DIGI:pdigi_valid,L1,DIGI2RAW,HLT:HIon,RAW2DIGI,L1Reco', '--inputCommands':'"keep *","drop *_simEcalPreshowerDigis_*_*"', '-n':2}, hiDefaults, step2Upg2015Defaults])



# PRE-MIXING : https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideSimulation#Pre_Mixing_Instructions
premixUp2015Defaults = {
    '--evt_type'    : 'SingleNuE10_cfi',
    '-s'            : 'GEN,SIM,DIGIPREMIX,L1,DIGI2RAW',
    '-n'            : '10',
    '--conditions'  : 'auto:run2_mc', # 25ns GT; dedicated dict for 50ns
    '--datatier'    : 'GEN-SIM-DIGI-RAW',
    '--eventcontent': 'PREMIX',
    '--magField'    : '38T_PostLS1',
    '--customise'   : 'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1' # temporary replacement for premix; to be brought back to customisePostLS1
}
premixUp2015Defaults50ns = merge([{'--conditions':'auto:run2_mc_50ns'},
                                  {'--customise':'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1_50ns'},
                                  premixUp2015Defaults])

steps['PREMIXUP15_PU25']=merge([PU25,Kby(100,100),premixUp2015Defaults])
steps['PREMIXUP15_PU50']=merge([PU50,Kby(100,100),premixUp2015Defaults50ns])

digiPremixUp2015Defaults25ns = { 
    '--conditions'   : 'auto:run2_mc',
    '-s'             : 'DIGIPREMIX_S2:pdigi_valid,DATAMIX,L1,DIGI2RAW,HLT:@frozen25ns,RAW2DIGI,L1Reco',
   '--pileup_input'  :  'das:/RelValPREMIXUP15_PU25/%s/GEN-SIM-DIGI-RAW'%baseDataSetRelease[5],
    '--eventcontent' : 'FEVTDEBUGHLT',
    '--datatier'     : 'GEN-SIM-DIGI-RAW-HLTDEBUG',
    '--datamix'      : 'PreMix',
    '--customise'    : 'SLHCUpgradeSimulations/Configuration/postLS1CustomsPreMixing.customisePostLS1', # temporary replacement for premix; to be brought back to customisePostLS1
    '--magField'     : '38T_PostLS1',
    }
digiPremixUp2015Defaults50ns=merge([{'-s':'DIGIPREMIX_S2:pdigi_valid,DATAMIX,L1,DIGI2RAW,HLT:@frozen50ns,RAW2DIGI,L1Reco'},
                                    {'--conditions':'auto:run2_mc_50ns'},
                                    {'--pileup_input' : 'das:/RelValPREMIXUP15_PU50/%s/GEN-SIM-DIGI-RAW'%baseDataSetRelease[6]},
                                    {'--customise': 'SLHCUpgradeSimulations/Configuration/postLS1CustomsPreMixing.customisePostLS1_50ns'},
                                    digiPremixUp2015Defaults25ns])
steps['DIGIPRMXUP15_PU25']=merge([digiPremixUp2015Defaults25ns])
steps['DIGIPRMXUP15_PU50']=merge([digiPremixUp2015Defaults50ns])
premixProd25ns = {'-s'             : 'DIGIPREMIX_S2,DATAMIX,L1,DIGI2RAW,HLT:@frozen25ns,RAW2DIGI,L1Reco',
                 '--eventcontent' : 'PREMIXRAW',
                 '--datatier'     : 'PREMIXRAW'}
premixProd50ns = merge([{'-s':'DIGIPREMIX_S2,DATAMIX,L1,DIGI2RAW,HLT:@frozen50ns,RAW2DIGI,L1Reco'},premixProd25ns])

steps['DIGIPRMXUP15_PROD_PU25']=merge([premixProd25ns,digiPremixUp2015Defaults25ns])
steps['DIGIPRMXUP15_PROD_PU50']=merge([premixProd50ns,digiPremixUp2015Defaults50ns])


dataReco={'--conditions':'auto:run1_data',
          '-s':'RAW2DIGI,L1Reco,RECO,EI,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+Hotline,DQM',
          '--datatier':'RECO,DQMIO',
          '--eventcontent':'RECO,DQM',
          '--data':'',
          '--process':'reRECO',
          '--scenario':'pp',
          }

dataRecoAlCaEle=merge([{'-s':'RAW2DIGI,L1Reco,RECO,EI,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalCalZElectron+EcalCalWElectron+EcalUncalZElectron+EcalUncalWElectron,DQM'}, dataReco])


hltKey='fake'
from Configuration.HLT.autoHLT import autoHLT
menu = autoHLT[hltKey]
steps['HLTD']=merge([{'--process':'reHLT',
                      '-s':'L1REPACK,HLT:@%s'%hltKey,
                      '--conditions':'auto:run1_hlt_%s'%menu,
                      '--data':'',
                      '--output':'\'[{"e":"RAW","t":"RAW","o":["drop FEDRawDataCollection_rawDataCollector__LHC"]}]\'',
                      },])
steps['HLTDSKIM']=merge([{'--inputCommands':'"keep *","drop *_*_*_RECO"'},steps['HLTD']])

steps['RECOD']=merge([{'--scenario':'pp',},dataReco])
steps['RECODAlCaEle']=merge([{'--scenario':'pp',},dataRecoAlCaEle])

steps['RECODSplit']=steps['RECOD'] # finer job splitting  
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
                         '--datatier':'RECO,DQMIO',
                         '--eventcontent':'RECO,DQM','-n':30},
                        steps['RECOD']])
steps['RECOHID11']=merge([{'--repacked':''},
                        steps['RECOHID10']])
steps['RECOHID10']['-s']+=',REPACK'
steps['RECOHID10']['--datatier']+=',RAW'
steps['RECOHID10']['--eventcontent']+=',REPACKRAW'

steps['TIER0']=merge([{'--customise':'Configuration/DataProcessing/RecoTLR.customisePrompt',
                       '-s':'RAW2DIGI,L1Reco,RECO,EI,ALCAPRODUCER:@allForPrompt,DQM,ENDJOB',
                       '--datatier':'RECO,AOD,ALCARECO,DQMIO',
                       '--eventcontent':'RECO,AOD,ALCARECO,DQM',
                       '--process':'RECO'
                       },dataReco])
steps['TIER0EXP']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,EI,ALCAPRODUCER:@allForExpress,DQM,ENDJOB',
                          '--datatier':'ALCARECO,DQMIO',
                          '--eventcontent':'ALCARECO,DQM',
                          '--customise':'Configuration/DataProcessing/RecoTLR.customiseExpress',
                          },steps['TIER0']])

steps['RECOCOSD']=merge([{'--scenario':'cosmics',
                          '-s':'RAW2DIGI,L1Reco,RECO,DQM,ALCA:MuAlCalIsolatedMu+DtCalib',
                          '--customise':'Configuration/DataProcessing/RecoTLR.customiseCosmicData'
                          },dataReco])

step2HImixDefaults=merge([{'-n':'2', #too slow for 10 events/hour
                           '--himix':'',
                           '--filein':'file.root',
                           '--process':'HISIGNAL'
                           },hiDefaults,step1Up2015Defaults])
steps['Pyquen_GammaJet_pt20_2760GeV']=merge([{'cfg':'Pyquen_GammaJet_pt20_2760GeV_cfi'},step2HImixDefaults])
steps['Pyquen_DiJet_pt80to120_2760GeV']=merge([{'cfg':'Pyquen_DiJet_pt80to120_2760GeV_cfi'},step2HImixDefaults])
steps['Pyquen_ZeemumuJets_pt10_2760GeV']=merge([{'cfg':'Pyquen_ZeemumuJets_pt10_2760GeV_cfi'},step2HImixDefaults])

# step3 
step3Defaults = {
                  '-s'            : 'RAW2DIGI,L1Reco,RECO,EI,VALIDATION,DQM',
                  '--conditions'  : 'auto:run1_mc',
                  '--no_exec'     : '',
                  '--datatier'    : 'GEN-SIM-RECO,DQMIO',
                  '--eventcontent': 'RECOSIM,DQM'
                  }
step3DefaultsAlCaEle=merge([{'-s':'RAW2DIGI,L1Reco,RECO,EI,ALCA:EcalCalZElectron+EcalCalWElectron+EcalUncalZElectron+EcalUncalWElectron,VALIDATION,DQM'}, step3Defaults])

steps['DIGIPU']=merge([{'--process':'REDIGI'},steps['DIGIPU1']])

#for 2015
step3Up2015Defaults = {'-s':'RAW2DIGI,L1Reco,RECO,EI,VALIDATION,DQM',
                 '--conditions':'auto:run2_mc', 
                 '--magField'    : '38T_PostLS1',
                 '-n':'10',
                 '--datatier':'GEN-SIM-RECO,DQMIO',
                 '--eventcontent':'RECOSIM,DQM',
                 '--customise' : 'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1'
                 }
step3Up2015Defaults50ns = merge([{'--conditions':'auto:run2_mc_50ns','--customise':'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1_50ns'},step3Up2015Defaults])

step3Up2015DefaultsAlCaEle = merge([{'-s':'RAW2DIGI,L1Reco,RECO,EI,ALCA:EcalCalZElectron+EcalCalWElectron+EcalUncalZElectron+EcalUncalWElectron,VALIDATION,DQM'},step3Up2015Defaults])
step3Up2015DefaultsAlCaEle50ns = merge([{'--conditions':'auto:run2_mc_50ns','--customise':'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1_50ns'},step3Up2015DefaultsAlCaEle])

step3Up2015Hal = {'-s'            :'RAW2DIGI,L1Reco,RECO,EI,VALIDATION,DQM',
                 '--conditions'   :'auto:run2_mc', 
                 '--magField'     :'38T_PostLS1',
                 '--datatier'     :'GEN-SIM-RECO,DQMIO',
                  '--eventcontent':'RECOSIM,DQM',
                  '-n'            :'10',
                 '--customise'    :'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1'
                 }

unSchOverrides={'--runUnscheduled':'','-s':'RAW2DIGI,L1Reco,RECO,EI,PAT,VALIDATION:@standardValidation+@miniAODValidation,DQM:@standardDQM+@miniAODDQM','--eventcontent':'RECOSIM,MINIAODSIM,DQM','--datatier':'GEN-SIM-RECO,MINIAODSIM,DQMIO'}
step3Up2015DefaultsUnsch = merge([unSchOverrides,step3Up2015Defaults])
step3DefaultsUnsch = merge([unSchOverrides,step3Defaults])

                             
steps['RECOUP15']=merge([step3Up2015Defaults]) # todo: remove UP from label
steps['RECOUP15AlCaEle']=merge([step3Up2015DefaultsAlCaEle]) # todo: remove UP from label

#steps['RECOUP15PROD1']=merge([{ '-s' : 'RAW2DIGI,L1Reco,RECO,EI,DQM:DQMOfflinePOGMC', '--datatier' : 'AODSIM,DQMIO', '--eventcontent' : 'AODSIM,DQM'},step3Up2015Defaults])

steps['RECODreHLT']=merge([{'--hltProcess':'reHLT','--conditions':'auto:run1_data_%s'%menu},steps['RECOD']])
steps['RECODreHLTAlCaEle']=merge([{'--hltProcess':'reHLT','--conditions':'auto:run1_data_%s'%menu},steps['RECODAlCaEle']])

steps['RECO']=merge([step3Defaults])
steps['RECOAlCaEle']=merge([step3DefaultsAlCaEle])
steps['RECODBG']=merge([{'--eventcontent':'RECODEBUG,DQM'},steps['RECO']])
steps['RECOPROD1']=merge([{ '-s' : 'RAW2DIGI,L1Reco,RECO,EI', '--datatier' : 'GEN-SIM-RECO,AODSIM', '--eventcontent' : 'RECOSIM,AODSIM'},step3Defaults])
steps['RECOPRODUP15']=merge([{ '-s':'RAW2DIGI,L1Reco,RECO,EI,DQM:DQMOfflinePOGMC','--datatier':'AODSIM,DQMIO','--eventcontent':'AODSIM,DQM'},step3Up2015Defaults])
steps['RECOCOS']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCA:MuAlCalIsolatedMu,DQM','--scenario':'cosmics'},stCond,step3Defaults])
steps['RECOHAL']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCA:MuAlCalIsolatedMu,DQM','--scenario':'cosmics'},step3Up2015Hal])
steps['RECOMIN']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,EI,ALCA:SiStripCalZeroBias+SiStripCalMinBias,VALIDATION,DQM'},stCond,step3Defaults])
steps['RECOMINUP15']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,EI,ALCA:SiStripCalZeroBias+SiStripCalMinBias,VALIDATION,DQM'},step3Up2015Defaults])

steps['RECODDQM']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,EI,DQM:@common+@muon+@hcal+@jetmet+@ecal'},steps['RECOD']])

steps['RECOPU1']=merge([PU,steps['RECO']])
steps['RECOPU2']=merge([PU2,steps['RECO']])
steps['RECOUP15_PU25']=merge([PU25,step3Up2015Defaults])
steps['RECOUP15_PU50']=merge([PU50,step3Up2015Defaults50ns])

#steps['RECOmAOD']=merge([step3DefaultsUnsch])
steps['RECOmAODUP15']=merge([step3Up2015DefaultsUnsch])


# for premixing: no --pileup_input for replay; GEN-SIM only available for in-time event, from FEVTDEBUGHLT previous step
steps['RECOPRMXUP15_PU25']=merge([
        {'-s':'RAW2DIGI,L1Reco,RECO,EI,VALIDATION,DQM'},
        {'--customise':'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1'}, # temporary replacement for premix; to be brought back to customisePostLS1
        step3Up2015Defaults])
steps['RECOPRMXUP15_PU50']=merge([
        {'-s':'RAW2DIGI,L1Reco,RECO,EI,VALIDATION,DQM'},
        {'--customise':'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1_50ns'},
        step3Up2015Defaults50ns])

recoPremixUp15prod = merge([
        {'-s':'RAW2DIGI,L1Reco,RECO,EI'},
        {'--datatier' : 'GEN-SIM-RECO,AODSIM'}, 
        {'--eventcontent' : 'RECOSIM,AODSIM'},
        {'--customise':'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1'}, # temporary replacement for premix; to be brought back to customisePostLS1
        step3Up2015Defaults])

steps['RECOPRMXUP15PROD_PU25']=merge([
        recoPremixUp15prod])
steps['RECOPRMXUP15PROD_PU50']=merge([
        {'--conditions':'auto:run2_mc_50ns'},
        {'--customise':'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1_50ns'},
        recoPremixUp15prod])


steps['RECOPUDBG']=merge([{'--eventcontent':'RECODEBUG,DQM'},steps['RECOPU1']])
steps['RERECOPU1']=merge([{'--hltProcess':'REDIGI'},steps['RECOPU1']])

steps['RECOUP15_ID']=merge([{'--hltProcess':'HLT2'},steps['RECOUP15']])

steps['RECOHI']=merge([hiDefaults,{'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM'},step3Up2015Defaults])
#wmsplit['RECOHI']=5

steps['DIGIHISt3']=steps['DIGIHI']

steps['RECOHID11St3']=merge([{
                              '--process':'ZStoRECO'},
                             steps['RECOHID11']])
steps['RECOHIR10D11']=merge([{'--filein':'file:step2_inREPACKRAW.root',
                              '--filtername':'reRECO'},
                             steps['RECOHID11St3']])
#steps['RECOFS']=merge([{'--fast':'',
#                        '-s':'RECO,EI,HLT:@fake,VALIDATION'},
#                       steps['RECO']])

#add this line when testing from an input file that is not strictly GEN-SIM
#addForAll(step3,{'--hltProcess':'DIGI'})

steps['ALCACOSD']={'--conditions':'auto:run1_data',
                   '--datatier':'ALCARECO',
                   '--eventcontent':'ALCARECO',
                   '--scenario':'cosmics',
                   '-s':'ALCA:TkAlCosmics0T+MuAlGlobalCosmics+HcalCalHOCosmics+DQM'
                   }
steps['ALCAPROMPT']={'-s':'ALCA:PromptCalibProd',
                     '--filein':'file:TkAlMinBias.root',
                     '--conditions':'auto:run1_data',
                     '--datatier':'ALCARECO',
                     '--eventcontent':'ALCARECO'}
steps['ALCAEXP']={'-s':'ALCA:PromptCalibProd',
                  '--conditions':'auto:run1_data',
                  '--datatier':'ALCARECO',
                  '--eventcontent':'ALCARECO'}

# step4
step4Defaults = { 
                  '-s'            : 'ALCA:TkAlMuonIsolated+TkAlMinBias+HcalCalIsoTrk+MuAlOverlaps',
                  '-n'            : 1000,
                  '--conditions'  : 'auto:run1_mc',
                  '--datatier'    : 'ALCARECO',
                  '--eventcontent': 'ALCARECO',
                  }
step4Up2015Defaults = { 
                        '-s'            : 'ALCA:TkAlMuonIsolated+TkAlMinBias+HcalCalIsoTrk+MuAlOverlaps',
                        '-n'            : 1000,
                        '--conditions'  : 'auto:run2_mc',
                        '--customise'   : 'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1',
                        '--datatier'    : 'ALCARECO',
                        '--eventcontent': 'ALCARECO',
                  }

steps['RERECOPU']=steps['RERECOPU1']

steps['ALCATT']=merge([{'--filein':'file:step3.root'},step4Defaults])
steps['ALCATTUp15']=merge([{'--filein':'file:step3.root'},step4Up2015Defaults])
steps['ALCAMIN']=merge([{'-s':'ALCA:TkAlMinBias','--filein':'file:step3.root'},stCond,step4Defaults])
steps['ALCACOS']=merge([{'-s':'ALCA:TkAlCosmics0T+MuAlGlobalCosmics+HcalCalHOCosmics'},stCond,step4Defaults])
steps['ALCABH']=merge([{'-s':'ALCA:TkAlBeamHalo+MuAlBeamHaloOverlaps+MuAlBeamHalo'},stCond,step4Defaults])
steps['ALCAHAL']=merge([{'-s':'ALCA:TkAlBeamHalo+MuAlBeamHaloOverlaps+MuAlBeamHalo'},step4Up2015Defaults])

steps['ALCAHARVD']={'-s':'ALCAHARVEST:BeamSpotByRun+BeamSpotByLumi+SiStripQuality',
                    '--conditions':'auto:run1_data',
                    '--scenario':'pp',
                    '--data':'',
                    '--filein':'file:PromptCalibProd.root'}

steps['RECOHISt4']=steps['RECOHI']

steps['ALCANZS']=merge([{'-s':'ALCA:HcalCalMinBias','--mc':''},step4Defaults])
steps['HARVESTGEN']={'-s':'HARVESTING:genHarvesting',
                     '--harvesting':'AtJobEnd',
                     '--conditions':'auto:run2_mc_FULL',
                     '--mc':'',
                     '--customise' :'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1',
                     '--filetype':'DQM',
                     '--filein':'file:step1_inDQM.root'
                  }

steps['HARVESTGEN2']=merge([{'--filein':'file:step2_inDQM.root'},steps['HARVESTGEN']]) 


#data
steps['HARVESTD']={'-s':'HARVESTING:dqmHarvesting',
                   '--conditions':'auto:run1_data',
                   '--data':'',
                   '--filetype':'DQM',
                   '--scenario':'pp'}

steps['HARVESTDreHLT'] = merge([ {'--conditions':'auto:run1_data_%s'%menu}, steps['HARVESTD'] ])

steps['HARVESTDDQM']=merge([{'-s':'HARVESTING:@common+@muon+@hcal+@jetmet+@ecal'},steps['HARVESTD']])

steps['HARVESTDfst2']=merge([{'--filein':'file:step2_inDQM.root'},steps['HARVESTD']])

steps['HARVESTDC']={'-s':'HARVESTING:dqmHarvesting',
                   '--conditions':'auto:run1_data',
                   '--filetype':'DQM',
                   '--data':'',
                    '--filein':'file:step2_inDQM.root',
                   '--scenario':'cosmics'}
steps['HARVESTDHI']={'-s':'HARVESTING:dqmHarvesting',
                   '--conditions':'auto:run1_data',
                   '--filetype':'DQM',
                   '--data':'',
                   '--scenario':'HeavyIons'}

#MC
steps['HARVEST']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                   '--conditions':'auto:run1_mc',
                   '--mc':'',
                   '--filetype':'DQM',
                   '--scenario':'pp'}
steps['HARVESTCOS']={'-s':'HARVESTING:dqmHarvesting',
                     '--conditions':'auto:run1_mc',
                     '--mc':'',
                     '--filein':'file:step3_inDQM.root',
                     '--filetype':'DQM',
                     '--scenario':'cosmics'}
steps['HARVESTHAL']={'-s'          :'HARVESTING:dqmHarvesting',
                     '--conditions':'auto:run2_mc',
                     '--magField'  :'38T_PostLS1',
                     '--mc'        :'',
                     '--filein'    :'file:step3_inDQM.root',
                     '--scenario'    :'cosmics',
                     '--filein':'file:step3_inDQM.root', # unnnecessary
                     '--filetype':'DQM',
                     '--customise' : 'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1',
                     }
steps['HARVESTFS']={'-s':'HARVESTING:validationHarvestingFS',
                   '--conditions':'auto:run1_mc',
                   '--mc':'',
                    '--filetype':'DQM',
                   '--scenario':'pp'}
steps['HARVESTHI']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                    '--conditions':'auto:run2_mc_HIon',
                    '--magField'    : '38T_PostLS1',
                    '--mc':'',
                    '--customise' : 'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1_HI',
                    '--filetype':'DQM',
                    '--scenario':'HeavyIons'}
steps['HARVESTUP15']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting', # todo: remove UP from label
                   '--conditions':'auto:run2_mc', 
                   '--magField'    : '38T_PostLS1',
                   '--mc':'',
                   '--customise' : 'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1',
                   '--filetype':'DQM',
                   }


steps['HARVESTUP15_PU25']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting', # todo: remove UP from label
                   '--conditions':'auto:run2_mc', 
                   '--magField'    : '38T_PostLS1',
                   '--mc':'',
                   '--customise' : 'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1',
                   '--filetype':'DQM',
                   }

steps['HARVESTUP15_PU50']={'-s':'HARVESTING:validationHarvesting+dqmHarvesting', # todo: remove UP from label
                   '--conditions':'auto:run2_mc_50ns', 
                   '--magField'    : '38T_PostLS1',
                   '--mc':'',
                   '--customise' : 'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1',
                   '--filetype':'DQM',
                   }

unSchHarvestOverrides={'-s':'HARVESTING:@standardValidation+@standardDQM+@miniAODValidation+@miniAODDQM'}
steps['HARVESTmAODUP15']=merge([unSchHarvestOverrides,steps['HARVESTUP15']])

steps['HARVESTUP15FS']={'-s':'HARVESTING:validationHarvestingFS',
                        '--conditions':'auto:run2_mc',
                        '--mc':'',
                        '--customise':'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1',
                        '--filetype':'DQM',
                        '--scenario':'pp'}


steps['ALCASPLIT']={'-s':'ALCAOUTPUT:@allForPrompt',
                    '--conditions':'auto:run1_data',
                    '--scenario':'pp',
                    '--data':'',
                    '--triggerResultsProcess':'RECO',
                    '--filein':'file:step2_inALCARECO.root'}

steps['SKIMD']={'-s':'SKIM:all',
                '--conditions':'auto:run1_data',
                '--data':'',
                '--scenario':'pp',
                '--filein':'file:step2.root',
                '--secondfilein':'filelist:step1_dasquery.log'}

steps['SKIMDreHLT'] = merge([ {'--conditions':'auto:run1_data_%s'%menu,'--filein':'file:step3.root'}, steps['SKIMD'] ])

steps['SKIMCOSD']={'-s':'SKIM:all',
                   '--conditions':'auto:run1_data',
                   '--data':'',
                   '--scenario':'cosmics',
                   '--filein':'file:step2.root',
                   '--secondfilein':'filelist:step1_dasquery.log'}
                 
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
                    '--conditions':'auto:run1_mc',
                    '--output':'\'[{"t":"RAW","e":"ALL"}]\'',
                    '--customise_commands':'"process.ALLRAWoutput.fastCloning=cms.untracked.bool(False)"'}

#miniaod
stepMiniAODDefaults = { '-s'              : 'PAT',
                        '--runUnscheduled': '',
                        '--customise'     : 'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1',
                        '-n'              : '100'
                        }
stepMiniAODDataUP15 = merge([{'--conditions'   : 'auto:run1_data',
                          '--data'         : '',
                          '--datatier'     : 'MINIAOD',
                          '--eventcontent' : 'MINIAOD',
                          '--filein'       :'file:step3.root'
                          },stepMiniAODDefaults])

stepMiniAODData = remove(stepMiniAODDataUP15,'--customise')

stepMiniAODMC = merge([{'--conditions'   : 'auto:run2_mc',
                        '--mc'           : '',
                        '--customise'    : 'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1',
                        '--datatier'     : 'MINIAODSIM',
                        '--eventcontent' : 'MINIAODSIM',
                        '--filein'       :'file:step3.root'
                        },stepMiniAODDefaults])

steps['MINIAODDATA']       =merge([stepMiniAODData])
steps['MINIAODDreHLT']     =merge([{'--conditions':'auto:run1_data_%s'%menu},stepMiniAODData])
steps['MINIAODDATAs2']     =merge([{'--filein':'file:step2.root'},stepMiniAODData])
steps['MINIAODMCUP15']     =merge([stepMiniAODMC])
steps['MINIAODMCUP1550']   =merge([{'--conditions':'auto:run2_mc_50ns','--customise':'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1_50ns'},stepMiniAODMC])
steps['MINIAODMCUP15HI']   =merge([{'--conditions':'auto:run2_mc_HIon','--customise':'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1_HI'},stepMiniAODMC])
steps['MINIAODMCUP15FS']   =merge([{'--filein':'file:step1.root','--fast':''},stepMiniAODMC])
steps['MINIAODMCUP15FS50'] =merge([{'--conditions':'auto:run2_mc_50ns','--customise':'SLHCUpgradeSimulations/Configuration/postLS1Customs.customisePostLS1_50ns'},steps['MINIAODMCUP15FS']])


#################################################################################
####From this line till the end of the file :
####UPGRADE WORKFLOWS IN PREPARATION - Gaelle's sandbox - 
#####Accessible only through the option --what upgrade
#####therefore not run in IBs (at some they might be...) 
#####Transparent for any of the standard workflows
#### list of worflows defined (not necessarly running though): runTheMatrix.py --what upgrade -n 
#### 
###
#################################################################################

from  Configuration.PyReleaseValidation.upgradeWorkflowComponents import *

defaultDataSets={}
defaultDataSets['Extended2023HGCalMuon']='CMSSW_6_2_0_SLHC20_patch1-DES23_62_V1_refHGCALV5-v'
defaultDataSets['Extended2023SHCalNoTaper']='CMSSW_6_2_0_SLHC20_patch1-DES23_62_V1_refSHNoTaper-v'
defaultDataSets['2019WithGEMAging']='CMSSW_6_2_0_SLHC20-DES19_62_V8_UPG2019withGEM-v'
keys=defaultDataSets.keys()
for key in keys:
  defaultDataSets[key+'PU']=defaultDataSets[key]
  
# sometimes v1 won't be used - override it here - the dictionary key is gen fragment + '_' + geometry
versionOverrides={}

baseDataSetReleaseBetter={}
for gen in upgradeFragments:
    for ds in defaultDataSets: 
       	key=gen[:-4]+'_'+ds   
        version='1'
        if key in versionOverrides:
            version = versionOverrides[key]
        baseDataSetReleaseBetter[key]=defaultDataSets[ds]+version

PUDataSets={}
for ds in defaultDataSets:
    key='MinBias_TuneZ2star_14TeV_pythia6'+'_'+ds
    name=baseDataSetReleaseBetter[key]
    PUDataSets[ds]={'-n':10,'--pileup':'AVE_140_BX_25ns','--pileup_input':'das:/RelValMinBias_TuneZ2star_14TeV/%s/GEN-SIM'%(name,)}


upgradeStepDict={}
for step in upgradeSteps:
    upgradeStepDict[step]={}

# just make all combinations - yes, some will be nonsense.. but then these are not used unless
# specified above
for k in upgradeKeys:
    k2=k
    if 'PU' in k[-2:]:
        k2=k[:-2]
    geom=upgradeGeoms[k2]
    gt=upgradeGTs[k2]
    cust=upgradeCustoms[k2]
    upgradeStepDict['GenSimFull'][k]= {'-s' : 'GEN,SIM',
                                       '-n' : 10,
                                       '--conditions' : gt,
                                       '--beamspot' : 'Gauss',
                                       '--magField' : '38T_PostLS1',
                                       '--datatier' : 'GEN-SIM',
                                       '--eventcontent': 'FEVTDEBUG',
                                       '--geometry' : geom
                                       }
    if cust!=None : upgradeStepDict['GenSimFull'][k]['--customise']=cust
        
    upgradeStepDict['GenSimHLBeamSpotFull'][k]= {'-s' : 'GEN,SIM',
                                       '-n' : 10,
                                       '--conditions' : gt,
                                       '--beamspot' : 'HLLHC',
                                       '--magField' : '38T_PostLS1',
                                       '--datatier' : 'GEN-SIM',
                                       '--eventcontent': 'FEVTDEBUG',
                                       '--geometry' : geom
                                       }
    if cust!=None : upgradeStepDict['GenSimHLBeamSpotFull'][k]['--customise']=cust
    
    upgradeStepDict['DigiFull'][k] = {'-s':'DIGI:pdigi_valid,L1,DIGI2RAW',
                                      '--conditions':gt,
                                      '--datatier':'GEN-SIM-DIGI-RAW',
                                      '-n':'10',
                                      '--magField' : '38T_PostLS1',
                                      '--eventcontent':'FEVTDEBUGHLT',
                                      '--geometry' : geom
                                      }
    if cust!=None : upgradeStepDict['DigiFull'][k]['--customise']=cust
    
    upgradeStepDict['DigiFullTrigger'][k] = {'-s':'DIGI:pdigi_valid,L1,L1TrackTrigger,DIGI2RAW',
                                      '--conditions':gt,
                                      '--datatier':'GEN-SIM-DIGI-RAW',
                                      '-n':'10',
                                      '--magField' : '38T_PostLS1',
                                      '--eventcontent':'FEVTDEBUGHLT',
                                      '--geometry' : geom
                                      }
    if cust!=None : upgradeStepDict['DigiFullTrigger'][k]['--customise']=cust
    
    
    if k2 in PUDataSets:
        upgradeStepDict['DigiFullPU'][k]=merge([PUDataSets[k2],upgradeStepDict['DigiFull'][k]])
    upgradeStepDict['DigiTrkTrigFull'][k] = {'-s':'DIGI:pdigi_valid,L1,L1TrackTrigger,DIGI2RAW,RECO:pixeltrackerlocalreco',
                                             '--conditions':gt,
                                             '--datatier':'GEN-SIM-DIGI-RAW',
                                             '-n':'10',
                                             '--magField' : '38T_PostLS1',
                                             '--eventcontent':'FEVTDEBUGHLT',
                                             '--geometry' : geom
                                             }
    if cust!=None : upgradeStepDict['DigiTrkTrigFull'][k]['--customise']=cust

    upgradeStepDict['RecoFull'][k] = {'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                                      '--conditions':gt,
                                      '--datatier':'GEN-SIM-RECO,DQMIO',
                                      '-n':'10',
                                      '--eventcontent':'FEVTDEBUGHLT,DQM',
                                      '--magField' : '38T_PostLS1',
                                      '--geometry' : geom
                                      }
    if cust!=None : upgradeStepDict['RecoFull'][k]['--customise']=cust

    if k2 in PUDataSets:
        upgradeStepDict['RecoFullPU'][k]=merge([PUDataSets[k2],{'-s':'RAW2DIGI,L1Reco,RECO,DQM'},upgradeStepDict['RecoFull'][k]])

    upgradeStepDict['RecoFullHGCAL'][k] = {'-s':'RAW2DIGI,L1Reco,RECO',
                                      '--conditions':gt,
                                      '--datatier':'GEN-SIM-RECO',
                                      '-n':'10',
                                      '--eventcontent':'RECOSIM',
                                      '--magField' : '38T_PostLS1',
                                      '--geometry' : geom
                                      }
    if cust!=None : upgradeStepDict['RecoFullHGCAL'][k]['--customise']=cust

    if k2 in PUDataSets:
        upgradeStepDict['RecoFullPUHGCAL'][k]=merge([PUDataSets[k2],{'-s':'RAW2DIGI,L1Reco,RECO'},upgradeStepDict['RecoFullHGCAL'][k]])

    upgradeStepDict['HARVESTFull'][k]={'-s':'HARVESTING:validationHarvesting+dqmHarvesting',
                                    '--conditions':gt,
                                    '--mc':'',
                                    '--magField' : '38T_PostLS1',
                                    '--geometry' : geom,
                                    '--scenario' : 'pp',
                                    '--filetype':'DQM'
                                    }
    if cust!=None : upgradeStepDict['HARVESTFull'][k]['--customise']=cust

    if k2 in PUDataSets:
        upgradeStepDict['HARVESTFullPU'][k]=merge([PUDataSets[k2],{'-s':'HARVESTING:dqmHarvesting'},upgradeStepDict['HARVESTFull'][k]])

    upgradeStepDict['FastSim'][k]={'-s':'GEN,SIM,RECO,VALIDATION',
                                   '--eventcontent':'FEVTDEBUGHLT,DQM',
                                   '--datatier':'GEN-SIM-DIGI-RECO,DQMIO',
                                   '--conditions':gt,
                                   '--fast':'',
                                   '--geometry' : geom,
                                   '--relval':'27000,3000'}
    if cust!=None : upgradeStepDict['FastSim'][k]['--customise']=cust

    upgradeStepDict['HARVESTFast'][k]={'-s':'HARVESTING:validationHarvestingFS',
                                    '--conditions':gt,
                                    '--mc':'',
                                    '--magField' : '38T_PostLS1',
                                    '--geometry' : geom,
                                    '--scenario' : 'pp'
                                    }
    if cust!=None : upgradeStepDict['HARVESTFast'][k]['--customise']=cust




for step in upgradeSteps:
    # we need to do this for each fragment
   if 'Sim' in step:
        for frag in upgradeFragments:
            howMuch=howMuches[frag]
            for key in upgradeKeys:
                k=frag[:-4]+'_'+key+'_'+step
                steps[k]=merge([ {'cfg':frag},howMuch,upgradeStepDict[step][key]])
                #get inputs in case of -i...but no need to specify in great detail
                #however, there can be a conflict of beam spots but this is lost in the dataset name
                #so please be careful   
                s=frag[:-4]+'_'+key
                if 'FastSim' not in k and s+'INPUT' not in steps and s in baseDataSetReleaseBetter:
                    steps[k+'INPUT']={'INPUT':InputInfo(dataSet='/RelVal'+frag[:-4]+'/%s/GEN-SIM'%(baseDataSetReleaseBetter[s],),location='STD')}
   else:
        for key in upgradeKeys:
            k=step+'_'+key
            if step in upgradeStepDict and key in upgradeStepDict[step]: 
                steps[k]=merge([upgradeStepDict[step][key]])
            
