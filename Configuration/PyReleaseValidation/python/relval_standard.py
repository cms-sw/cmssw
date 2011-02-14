
class InputInfo(object):
    def __init__(self) :
        self.run = 1
        self.files = 1
        self.events = 1
        self.location = 'STD'


# define input info objects for the data inputs
real2010Av1 = InputInfo()
real2010Av1.dataSet  = 'Cosmics/Run2010A-v1/RAW',
real2010Av1.run      = 142089,
real2010Av1.files    = 100,
real2010Av1.events   = 100000,
real2010Av1.label    = 'cos2010A',
real2010Av1.location = 'CAF',  

minBiasRelVal = InputInfo()
minBiasRelVal.dataSet  = '/RelValProdMinBias/CMSSW_3_11_0_pre5-MC_311_V0-v1/GEN-SIM-RAW'
minBiasRelVal.label    = 'minbiasrv'

ttBarRelVal = InputInfo()
ttBarRelVal.dataSet  = '/RelValTTbar/CMSSW_3_11_0_pre5-START311_V0-v1/GEN-SIM-DIGI-RAW-HLTDEBUG'
ttBarRelVal.label    = 'ttbarrv'


# step1 gensim
step1Defaults = {'cfg'           : None, # need to be explicitly set
                 'INPUT'         : None, # use this if you want to read from file instead, leave it as None to use simgen
                 '--relval'      : None, # need to be explicitly set
                 '-s'            : 'GEN,SIM',
                 '-n'            : 10,
                 '--geometry'    : 'DB',
                 '--conditions'  : 'auto:mc',
                 '--datatier'    : 'GEN-SIM',
                 '--eventcontent': 'RAWSIM',
                 }


step1 = { 'ProdMinBias'      : { 'cfg' : 'MinBias_7T3V.cfi'     , '--relval' : '9000,100'},
          'ProdMinBiasINPUT' : { 'INPUT' : minBiasRelVal, '--relval' : '9000,100'},
          'TTbar'            : { 'cfg' : 'TTbar_Tauola_7TeV.cfi', '--relval' : '9000, 50'},
          'TTbarINPUT'       : { 'INPUT' : ttBarRelVal  , '--relval' : '9000, 50'},
          'ProdQCD_Pt_3000_3500' : { 'cfg' : 'QCD_Pt_3000_3500_7TeV.cfi',  '-s' : 'GEN,SIM,DIGI,L1,DIGI2RAW,HLT:GRun,RAW2DIGI,L1Reco', '--relval': '9000,25', '--datatier' : 'GEN-SIM-RAW'},
         }


# step2 reco
step2Defaults = { 'cfg'           : 'step2',
                  '-s'            : 'DIGI,L1,DIGI2RAW,HLT:GRun,RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                  '--datatier'    : 'GEN-SIM-RECO,AODSIM,DQM',
                  '--eventcontent': 'RECOSIM,AODSIM,DQM',
                  '--geometry'    : 'DB',
                  '--conditions'  : 'auto:mc',
                  }

step2 = {'HLTRECO1' : { '--conditions' : 'auto:mc'    },
         'HLTRECO2' : { '--conditions' : 'auto:start' },
         'RECOPROD1': { '-s' : 'RAW2DIGI,L1Reco,RECO', '--datatier' : 'GEN-SIM-RECO,AODSIM', '--eventcontent' : 'RECOSIM,AODSIM'},
         }


# step3 - alca 
step3Defaults = { 'cfg'           : 'step3_RELVAL',
                  '-s'            : 'ALCA:TkAlMuonIsolated+TkAlMinBias+EcalCalElectron+HcalCalIsoTrk+MuAlOverlaps',
                  '-n'            : 1000,
                  '--filein'      : 'file:reco.root',
                  '--geometry'    : 'DB',
                  '--conditions'  : 'auto:mc',
                  '--no_exec'     : '',
                  '--datatier'    : 'ALCARECO',
                  '--oneoutput'   : '',
                  '--eventcontent': 'ALCARECO',
                  }

step3 = { 'ALCATT1' : { '--conditions' : 'auto:mc'    },
          'ALCATT2' : { '--conditions' : 'auto:start' },
          }


# nothing defined yet, but keep in list for compatibility
step4Defaults = {}
step4 = {}

# for easier looping: (or should each step have a "default" item ??)
stepDefaultsList = [step1Defaults, step2Defaults, step3Defaults, step4Defaults]
stepList = [step1, step2, step3, step4]

steps = zip(stepList, stepDefaultsList)


# finally define the workflows as a combination of the above:
workflows = {  1   : ['ProdMinBias', 'HLTRECO1'],
               1.1 : ['ProdMinBias', 'HLTRECO2'],
               3.0 : ['ProdQCD_Pt_3000_3500', 'RECOPROD1'],
              24   : ['TTbar', 'HLTRECO1'],
              25   : ['TTbar', 'HLTRECO2', 'ALCATT2'],
              }


