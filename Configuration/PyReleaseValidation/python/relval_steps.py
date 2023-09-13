from __future__ import absolute_import
from .MatrixUtil import *

from Configuration.HLT.autoHLT import autoHLT
from Configuration.AlCa.autoPCL import autoPCL
from Configuration.Skimming.autoSkim import autoSkim
from .upgradeWorkflowComponents import step3_trackingOnly

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
                             '--beamspot'    : 'Realistic25ns13TeV2016Collision',
                             '--datatier'    : 'GEN-SIM',
                             '--eventcontent': 'FEVTDEBUG',
                             '--era'         : 'Run2_2016'
                             }
# step1 gensim: for 2017
step1Up2017Defaults = merge ([{'--conditions':'auto:phase1_2017_realistic','--era':'Run2_2017','--beamspot':'Realistic25ns13TeVEarly2017Collision'},step1Up2015Defaults])
# step1 gensim: for 2018
step1Up2018Defaults = merge ([{'--conditions':'auto:phase1_2018_realistic','--era':'Run2_2018','--beamspot':'Realistic25ns13TeVEarly2018Collision','--geometry':'DB:Extended'},step1Up2017Defaults])
# step1 gensim: for 2018 HI
step1Up2018HiDefaults = merge ([{'--conditions':'auto:phase1_2018_realistic_hi','--era':'Run2_2018_pp_on_AA','--beamspot':'RealisticPbPbCollision2018','--geometry':'DB:Extended'},step1Up2017Defaults])
# step1 gensim: for 2018 prod
step1Up2018ProdDefaults = merge ([{'--eventcontent':'RAWSIM'},step1Up2018Defaults])
# step1 gensim: for 2018 HI prod
step1Up2018HiProdDefaults = merge ([{'--eventcontent':'RAWSIM'},step1Up2018HiDefaults])
# step1 gensim: for 2022 HI prod
step1Up2022HiProdDefaults = merge ([{'--conditions':'auto:phase1_2022_realistic_hi','--era':'Run3_pp_on_PbPb','--beamspot':'Realistic2022PbPbCollision','--eventcontent':'RAWSIM','--geometry':'DB:Extended'},step1Up2018HiDefaults])
step1Up2023HiProdDefaults = merge ([{'--conditions':'auto:phase1_2023_realistic_hi','--era':'Run3_pp_on_PbPb_2023','--beamspot':'Realistic2022PbPbCollision','--eventcontent':'RAWSIM','--geometry':'DB:Extended'},step1Up2018HiDefaults])

steps = Steps()

#### Production test section ####
steps['ProdMinBias']=merge([{'cfg':'MinBias_8TeV_pythia8_TuneCUETP8M1_cff','--relval':'9000,300'},step1Defaults])
steps['ProdTTbar']=merge([{'cfg':'TTbar_8TeV_TuneCUETP8M1_cfi','--relval':'9000,100'},step1Defaults])
steps['ProdQCD_Pt_3000_3500']=merge([{'cfg':'QCD_Pt_3000_3500_8TeV_TuneCUETP8M1_cfi','--relval':'9000,50'},step1Defaults])

#### data ####
Run2010ASk=[138937,138934,138924,138923,139790,139789,139788,139787,144086,144085,144084,144083,144011]
Run2010BSk=[146644,147115,147929,148822,149011,149181,149182,149291,149294,149442]
steps['MinimumBias2010A']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2010A-valskim-v6/RAW-RECO',label='2010A',location='STD',run=Run2010ASk)}
steps['MinimumBias2010B']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2010B-valskim-v2/RAW-RECO',label='2010B',run=Run2010BSk)}
steps['WZMuSkim2010A']={'INPUT':InputInfo(dataSet='/Mu/Run2010A-WZMu-Nov4Skim_v1/RAW-RECO',label='2010A',run=Run2010ASk)}
steps['WZMuSkim2010B']={'INPUT':InputInfo(dataSet='/Mu/Run2010B-WZMu-Nov4Skim_v1/RAW-RECO',label='2010B',run=Run2010BSk)}
steps['WZEGSkim2010A']={'INPUT':InputInfo(dataSet='/EG/Run2010A-WZEG-Nov4Skim_v1/RAW-RECO',label='2010A',run=Run2010ASk)}
steps['WZEGSkim2010B']={'INPUT':InputInfo(dataSet='/Electron/Run2010B-WZEG-Nov4Skim_v1/RAW-RECO',label='2010B',run=Run2010BSk)}

steps['RunCosmicsA']={'INPUT':InputInfo(dataSet='/Cosmics/Run2010A-v1/RAW',label='2010A',run=[142089],events=100000)}
Run2010B=[149011]
steps['RunMinBias2010B']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2010B-RelValRawSkim-v1/RAW',label='2010B',run=Run2010B,events=100000)}
steps['RunMu2010B']={'INPUT':InputInfo(dataSet='/Mu/Run2010B-RelValRawSkim-v1/RAW',label='2010B',run=Run2010B,events=100000)}
steps['RunElectron2010B']={'INPUT':InputInfo(dataSet='/Electron/Run2010B-RelValRawSkim-v1/RAW',label='2010B',run=Run2010B,events=100000)}
steps['RunPhoton2010B']={'INPUT':InputInfo(dataSet='/Photon/Run2010B-RelValRawSkim-v1/RAW',label='2010B',run=Run2010B,events=100000)}
steps['RunJet2010B']={'INPUT':InputInfo(dataSet='/Jet/Run2010B-RelValRawSkim-v1/RAW',label='2010B',run=Run2010B,events=100000)}

#list of run to harvest 2011A: 165121, 172802,
Run2011ASk=[165121,172802]
steps['ValSkim2011A']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2011A-ValSkim-08Nov2011-v1/RAW-RECO',ib_block='239c497e-0fae-11e1-a8b1-00221959e72f',label='2011A',location='STD',run=Run2011ASk)}
steps['WMuSkim2011A']={'INPUT':InputInfo(dataSet='/SingleMu/Run2011A-WMu-08Nov2011-v1/RAW-RECO',ib_block='388c2990-0de6-11e1-bb7e-00221959e72f',label='2011A',location='STD',run=Run2011ASk)}
steps['WElSkim2011A']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2011A-WElectron-08Nov2011-v1/RAW-RECO',ib_block='9c48c4ea-0db2-11e1-b62c-00221959e69e',label='2011A',location='STD',run=Run2011ASk)}
steps['ZMuSkim2011A']={'INPUT':InputInfo(dataSet='/DoubleMu/Run2011A-ZMu-08Nov2011-v1/RAW-RECO',label='2011A',location='STD',run=Run2011ASk)}
steps['ZElSkim2011A']={'INPUT':InputInfo(dataSet='/DoubleElectron/Run2011A-ZElectron-08Nov2011-v1/RAW-RECO',label='2011A',location='STD',run=Run2011ASk)}
steps['HighMet2011A']={'INPUT':InputInfo(dataSet='/Jet/Run2011A-HighMET-08Nov2011-v1/RAW-RECO',ib_block='3c764584-0b59-11e1-b62c-00221959e69e',label='2011A',location='STD',run=Run2011ASk)}

steps['RunCosmics2011A']={'INPUT':InputInfo(dataSet='/Cosmics/Run2011A-v1/RAW',label='2011A',run=[160960],events=100000,location='STD')}
Run2011A=[165121]
Run2017BTE=[299149]
Run2016HALP=[283383]
steps['RunMinBias2011A']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2011A-v1/RAW',label='2011A',run=Run2011A,events=100000,location='STD')}
steps['RunMu2011A']={'INPUT':InputInfo(dataSet='/SingleMu/Run2011A-v1/RAW',label='2011A',run=Run2011A,events=100000)}
steps['RunElectron2011A']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2011A-v1/RAW',label='2011A',run=Run2011A,events=100000)}
steps['RunPhoton2011A']={'INPUT':InputInfo(dataSet='/Photon/Run2011A-v1/RAW',label='2011A',run=Run2011A,events=100000)}
steps['RunJet2011A']={'INPUT':InputInfo(dataSet='/Jet/Run2011A-v1/RAW',label='2011A',run=Run2011A,events=100000)}
steps['TestEnableEcalHCAL2017B']={'INPUT':InputInfo(dataSet='/TestEnablesEcalHcal/Run2017B-v1/RAW',label='2017B',run=Run2017BTE,events=100000,location='STD')}

Run2011B=[177719]
Run2011BSk=[177719,177790,177096,175874]
steps['RunMinBias2011B']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2011B-v1/RAW',label='2011B',run=Run2011B,events=100000,location='STD')}
steps['RunMu2011B']={'INPUT':InputInfo(dataSet='/SingleMu/Run2011B-v1/RAW',label='2011B',run=Run2011B,events=100000)}
steps['RunElectron2011B']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2011B-v1/RAW',label='2011B',run=Run2011B,events=100000)}
steps['RunPhoton2011B']={'INPUT':InputInfo(dataSet='/Photon/Run2011B-v1/RAW',label='2011B',run=Run2011B,events=100000)}
steps['RunJet2011B']={'INPUT':InputInfo(dataSet='/Jet/Run2011B-v1/RAW',label='2011B',run=Run2011B,events=100000)}

steps['ValSkim2011B']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2011B-ValSkim-19Nov2011-v1/RAW-RECO',label='2011B',location='STD',run=Run2011BSk)}
steps['WMuSkim2011B']={'INPUT':InputInfo(dataSet='/SingleMu/Run2011B-WMu-19Nov2011-v1/RAW-RECO',ib_block='19110c74-1b66-11e1-a98b-003048f02c8a',label='2011B',location='STD',run=Run2011BSk)}
steps['WElSkim2011B']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2011B-WElectron-19Nov2011-v1/RAW-RECO',ib_block='d75771a4-1b3f-11e1-aef4-003048f02c8a',label='2011B',location='STD',run=Run2011BSk)}
steps['ZMuSkim2011B']={'INPUT':InputInfo(dataSet='/DoubleMu/Run2011B-ZMu-19Nov2011-v1/RAW-RECO',label='2011B',location='STD',run=Run2011BSk)}
steps['ZElSkim2011B']={'INPUT':InputInfo(dataSet='/DoubleElectron/Run2011B-ZElectron-19Nov2011-v1/RAW-RECO',label='2011B',run=Run2011BSk)}
steps['HighMet2011B']={'INPUT':InputInfo(dataSet='/Jet/Run2011B-HighMET-19Nov2011-v1/RAW-RECO',label='2011B',run=Run2011BSk)}

steps['RunHI2010']={'INPUT':InputInfo(dataSet='/HIAllPhysics/HIRun2010-v1/RAW',label='hi2010',run=[152698],events=10000,location='STD')}
steps['RunHI2011']={'INPUT':InputInfo(dataSet='/HIMinBiasUPC/HIRun2011-v1/RAW',label='hi2011',ls={182124: [40]},events=10000,location='STD')}
steps['RunPA2013']={'INPUT':InputInfo(dataSet='/PAMinBiasUPC/HIRun2013-v1/RAW',label='pa2013',run=[211313],events=10000,location='STD')}

Run2015HI={263400: [[65,904]]}
steps['RunHI2015VR']={'INPUT':InputInfo(dataSet='/HITrackerVirginRaw/HIRun2015-v1/RAW',label='hi2015vr',events=10000,location='STD',ls=Run2015HI)}

#steps['RunHI2018']={'INPUT':InputInfo(dataSet='/HIMinimumBias0/Tier0_REPLAY_vocms015-v214/RAW',label='hi2018',run=[325174],events=10000,location='STD')}
#steps['RunHI2018Reduced']={'INPUT':InputInfo(dataSet='/HIMinimumBiasReducedFormat0/Tier0_REPLAY_vocms015-v214/RAW',label='hi2018reduced',run=[325174],events=10000,location='STD')}

Run2018HI={326479: [[1,23]]}
steps['RunHI2018']={'INPUT':InputInfo(dataSet='/HIHardProbes/HIRun2018A-v1/RAW',label='hi2018',events=10000,location='STD',ls=Run2018HI)}
steps['RunHI2018Reduced']={'INPUT':InputInfo(dataSet='/HIMinimumBiasReducedFormat0/HIRun2018A-v1/RAW',label='hi2018reduced',events=10000,location='STD',ls=Run2018HI)}
steps['RunHI2018AOD']={'INPUT':InputInfo(dataSet='/HIHardProbes/HIRun2018A-04Apr2019-v1/AOD',label='hi2018aod',events=10000,location='STD',ls=Run2018HI)}

Run2022HI={362321: [[112,112]]}
steps['RunHI2022']={'INPUT':InputInfo(dataSet='/HITestRaw0/HIRun2022A-v1/RAW',label='hi2022',events=10000,location='STD',ls=Run2022HI)}

Run2012A=[191226]
Run2012ASk=Run2012A+[]
steps['RunMinBias2012A']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2012A-v1/RAW',label='2012A',run=Run2012A, events=100000,location='STD')}
steps['RunTau2012A']={'INPUT':InputInfo(dataSet='/Tau/Run2012A-v1/RAW',label='2012A', run=Run2012A, events=100000,location='STD')}
steps['RunMET2012A']={'INPUT':InputInfo(dataSet='/MET/Run2012A-v1/RAW',label='2012A', run=Run2012A, events=100000,location='STD')}
steps['RunMu2012A']={'INPUT':InputInfo(dataSet='/SingleMu/Run2012A-v1/RAW',label='2012A', run=Run2012A, events=100000,location='STD')}
steps['RunElectron2012A']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2012A-v1/RAW',label='2012A', run=Run2012A, events=100000,location='STD')}
steps['RunJet2012A']={'INPUT':InputInfo(dataSet='/Jet/Run2012A-v1/RAW',label='2012A', run=Run2012A, events=100000,location='STD')}

steps['WElSkim2012A']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2012A-WElectron-13Jul2012-v1/USER',label='2012A',location='STD',run=Run2012ASk)}
steps['ZMuSkim2012A']={'INPUT':InputInfo(dataSet='/SingleMu/Run2012A-ZMu-13Jul2012-v1/RAW-RECO',label='2012A',location='STD',run=Run2012ASk)}
steps['ZElSkim2012A']={'INPUT':InputInfo(dataSet='/DoubleElectron/Run2012A-ZElectron-13Jul2012-v1/RAW-RECO',label='2012A',run=Run2012ASk)}
steps['HighMet2012A']={'INPUT':InputInfo(dataSet='/HT/Run2012A-HighMET-13Jul2012-v1/RAW-RECO',label='2012A',run=Run2012ASk)}


Run2012B=[194533]
Run2012Bsk=Run2012B+[194912,195016]
steps['RunMinBias2012B']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2012B-v1/RAW',label='2012B',run=Run2012B, events=100000,location='STD')}
steps['RunMu2012B']={'INPUT':InputInfo(dataSet='/SingleMu/Run2012B-v1/RAW',label='2012B',location='STD',run=Run2012B)}
steps['RunPhoton2012B']={'INPUT':InputInfo(dataSet='/SinglePhoton/Run2012B-v1/RAW',ib_block='28d7fcc8-a2a0-11e1-86c7-003048caaace',label='2012B',location='STD',run=Run2012B)}
steps['RunEl2012B']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2012B-v1/RAW',label='2012B',location='STD',run=Run2012B)}
steps['RunJet2012B']={'INPUT':InputInfo(dataSet='/JetHT/Run2012B-v1/RAW',label='2012B',location='STD',run=Run2012B)}
steps['ZMuSkim2012B']={'INPUT':InputInfo(dataSet='/SingleMu/Run2012B-ZMu-13Jul2012-v1/RAW-RECO',label='2012B',location='CAF',run=Run2012Bsk)}
steps['WElSkim2012B']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2012B-WElectron-13Jul2012-v1/USER',label='2012B',location='STD',run=Run2012Bsk)}
steps['ZElSkim2012B']={'INPUT':InputInfo(dataSet='/DoubleElectron/Run2012B-ZElectron-22Jan2013-v1/RAW-RECO',ib_block='1f13b876-69fb-11e2-a7eb-00221959e72f',label='2012B',location='STD',run=Run2012Bsk)}

Run2012C=[199812]
Run2012Csk=Run2012C+[203002]
steps['RunMinBias2012C']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2012C-v1/RAW',label='2012C',run=Run2012C, events=100000,location='STD')}
steps['RunMu2012C']={'INPUT':InputInfo(dataSet='/SingleMu/Run2012C-v1/RAW',label='2012C',location='STD',run=Run2012C)}
steps['RunPhoton2012C']={'INPUT':InputInfo(dataSet='/SinglePhoton/Run2012C-v1/RAW',label='2012C',location='STD',run=Run2012C)}
steps['RunEl2012C']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2012C-v1/RAW',label='2012C',location='STD',run=Run2012C)}
steps['RunJet2012C']={'INPUT':InputInfo(dataSet='/JetHT/Run2012C-v1/RAW',label='2012C',location='STD',run=Run2012C)}
steps['ZMuSkim2012C']={'INPUT':InputInfo(dataSet='/SingleMu/Run2012C-ZMu-PromptSkim-v3/RAW-RECO',label='2012C',location='CAF',run=Run2012Csk)}
steps['WElSkim2012C']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2012C-WElectron-PromptSkim-v3/USER',label='2012C',location='STD',run=Run2012Csk)}
steps['ZElSkim2012C']={'INPUT':InputInfo(dataSet='/DoubleElectron/Run2012C-ZElectron-22Jan2013-v1/RAW-RECO',label='2012C',location='STD',run=Run2012Csk)}

Run2012D=[208307]
Run2012Dsk=Run2012D+[207454]
steps['RunMinBias2012D']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2012D-v1/RAW',label='2012D',run=Run2012D, events=100000,location='STD')}
steps['RunMu2012D']={'INPUT':InputInfo(dataSet='/SingleMu/Run2012D-v1/RAW',label='2012D',location='STD',run=Run2012D)}
steps['RunPhoton2012D']={'INPUT':InputInfo(dataSet='/SinglePhoton/Run2012D-v1/RAW',label='2012D',location='STD',run=Run2012D)}
steps['RunEl2012D']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2012D-v1/RAW',label='2012D',location='STD',run=Run2012D)}
steps['RunJet2012D']={'INPUT':InputInfo(dataSet='/JetHT/Run2012D-v1/RAW',label='2012D',location='STD',run=Run2012D)}
# the previous  /SingleMu/Run2012D-ZMu-15Apr2014-v1/RAW-RECO is deprecated in DAS
steps['ZMuSkim2012D']={'INPUT':InputInfo(dataSet='/SingleMu/Run2012D-ZMu-15Apr2014-v1/RAW-RECO',label='2012D',location='STD',run=Run2012Dsk)}
# the previous /SingleElectron/Run2012D-WElectron-PromptSkim-v1/USER is deprecated in DAS
steps['WElSkim2012D']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2012D-WElectron-22Jan2013-v1/USER',label='2012D',location='STD',run=Run2012Dsk)}
steps['ZElSkim2012D']={'INPUT':InputInfo(dataSet='/DoubleElectron/Run2012D-ZElectron-22Jan2013-v1/RAW-RECO',label='2012D',location='STD',run=Run2012Dsk)}

#### run2 2015B ####
# Run2015B=[251642] #  251561 251638 251642
Run2015B=selectedLS([251251])
steps['RunHLTPhy2015B']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2015B-v1/RAW',label='2015B',events=100000,location='STD', ls=Run2015B)}
steps['RunDoubleEG2015B']={'INPUT':InputInfo(dataSet='/DoubleEG/Run2015B-v1/RAW',label='2015B',events=100000,location='STD', ls=Run2015B)}
steps['RunDoubleMuon2015B']={'INPUT':InputInfo(dataSet='/DoubleMuon/Run2015B-v1/RAW',label='2015B',events=100000,location='STD', ls=Run2015B)}
steps['RunJetHT2015B']={'INPUT':InputInfo(dataSet='/JetHT/Run2015B-v1/RAW',label='2015B',events=100000,location='STD', ls=Run2015B)}
steps['RunMET2015B']={'INPUT':InputInfo(dataSet='/MET/Run2015B-v1/RAW',label='2015B',events=100000,location='STD', ls=Run2015B)}
steps['RunMuonEG2015B']={'INPUT':InputInfo(dataSet='/MuonEG/Run2015B-v1/RAW',label='2015B',events=100000,location='STD', ls=Run2015B)}
steps['RunSingleEl2015B']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2015B-v1/RAW',label='2015B',events=100000,location='STD', ls=Run2015B)}
steps['RunSingleMu2015B']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2015B-v1/RAW',label='2015B',events=100000,location='STD', ls=Run2015B)}
steps['RunSinglePh2015B']={'INPUT':InputInfo(dataSet='/SinglePhoton/Run2015B-v1/RAW',label='2015B',events=100000,location='STD', ls=Run2015B)}
steps['RunZeroBias2015B']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2015B-v1/RAW',label='2015B',events=100000,location='STD', ls=Run2015B)}




#### run2 2015C ####
# Run2015C, 25ns: 254790 (852 LS and 65 files), 254852 (126 LS and 5 files), 254879 (178 LS and 11 files)
Run2015C=selectedLS([254790])
Run2015C_full=selectedLS([254790, 254852, 254879])
steps['RunHLTPhy2015C']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2015C-v1/RAW',label='2015C',events=100000,location='STD', ls=Run2015C)}
steps['RunDoubleEG2015C']={'INPUT':InputInfo(dataSet='/DoubleEG/Run2015C-v1/RAW',label='2015C',events=100000,location='STD', ls=Run2015C)}
steps['RunDoubleMuon2015C']={'INPUT':InputInfo(dataSet='/DoubleMuon/Run2015C-v1/RAW',label='2015C',events=100000,location='STD', ls=Run2015C)}
steps['RunJetHT2015C']={'INPUT':InputInfo(dataSet='/JetHT/Run2015C-v1/RAW',label='2015C',events=100000,location='STD', ls=Run2015C)}
steps['RunMET2015C']={'INPUT':InputInfo(dataSet='/MET/Run2015C-v1/RAW',label='2015C',events=100000,location='STD', ls=Run2015C)}
steps['RunMuonEG2015C']={'INPUT':InputInfo(dataSet='/MuonEG/Run2015C-v1/RAW',label='2015C',events=100000,location='STD', ls=Run2015C)}
steps['RunDoubleEGPrpt2015C']={'INPUT':InputInfo(dataSet='/DoubleEG/Run2015C-ZElectron-PromptReco-v1/RAW-RECO',label='2015C',events=100000,location='STD', ls=Run2015C_full)}
steps['RunSingleMuPrpt2015C']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2015C-ZMu-PromptReco-v1/RAW-RECO',label='2015C',events=100000,location='STD', ls=Run2015C_full)}
steps['RunSingleEl2015C']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2015C-v1/RAW',label='2015C',events=100000,location='STD', ls=Run2015C)}
steps['RunSingleMu2015C']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2015C-v1/RAW',label='2015C',events=100000,location='STD', ls=Run2015C)}
steps['RunSinglePh2015C']={'INPUT':InputInfo(dataSet='/SinglePhoton/Run2015C-v1/RAW',label='2015C',events=100000,location='STD', ls=Run2015C)}
steps['RunZeroBias2015C']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2015C-v1/RAW',label='2015C',events=100000,location='STD', ls=Run2015C)}


#### run2 2015D ####
# Run2015D, 25ns: 256677
Run2015D=selectedLS([256677])
steps['RunHLTPhy2015D']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2015D-v1/RAW',label='2015D',events=100000,location='STD', ls=Run2015D)}
steps['RunDoubleEG2015D']={'INPUT':InputInfo(dataSet='/DoubleEG/Run2015D-v1/RAW',label='2015D',events=100000,location='STD', ls=Run2015D)}
steps['RunDoubleMuon2015D']={'INPUT':InputInfo(dataSet='/DoubleMuon/Run2015D-v1/RAW',label='2015D',events=100000,location='STD', ls=Run2015D)}
steps['RunJetHT2015D']={'INPUT':InputInfo(dataSet='/JetHT/Run2015D-v1/RAW',label='2015D',events=100000,location='STD', ls=Run2015D)}
steps['RunMET2015D']={'INPUT':InputInfo(dataSet='/MET/Run2015D-v1/RAW',label='2015D',events=100000,location='STD', ls=Run2015D)}
steps['RunMuonEG2015D']={'INPUT':InputInfo(dataSet='/MuonEG/Run2015D-v1/RAW',label='2015D',events=100000,location='STD', ls=Run2015D)}
steps['RunDoubleEGPrpt2015D']={'INPUT':InputInfo(dataSet='/DoubleEG/Run2015D-ZElectron-PromptReco-v3/RAW-RECO',label='2015D',events=100000,location='STD', ls=Run2015D)}
steps['RunSingleMuPrpt2015D']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2015D-ZMu-PromptReco-v3/RAW-RECO',label='2015D',events=100000,location='STD', ls=Run2015D)}
steps['RunSingleEl2015D']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2015D-v1/RAW',label='2015D',events=100000,location='STD', ls=Run2015D)}
steps['RunSingleMu2015D']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2015D-v1/RAW',label='2015D',events=100000,location='STD', ls=Run2015D)}
steps['RunSinglePh2015D']={'INPUT':InputInfo(dataSet='/SinglePhoton/Run2015D-v1/RAW',label='2015D',events=100000,location='STD', ls=Run2015D)}
steps['RunZeroBias2015D']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2015D-v1/RAW',label='2015D',events=100000,location='STD',ib_block='38d4cab6-5d5f-11e5-824b-001e67ac06a0',ls=Run2015D)}

#### run2 2016B ####
# Run2016B, 25ns: 274160
#Run2016B=selectedLS([274160],l_json=data_json2016)
Run2016B={274199: [[1, 180]]}
steps['RunHLTPhy2016B']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2016B-v2/RAW',label='2016B',events=100000,location='STD', ls=Run2016B)}
steps['RunDoubleEG2016B']={'INPUT':InputInfo(dataSet='/DoubleEG/Run2016B-v2/RAW',label='2016B',events=100000,location='STD', ls=Run2016B)}
steps['RunDoubleMuon2016B']={'INPUT':InputInfo(dataSet='/DoubleMuon/Run2016B-v2/RAW',label='2016B',events=100000,location='STD', ls=Run2016B)}
steps['RunJetHT2016B']={'INPUT':InputInfo(dataSet='/JetHT/Run2016B-v2/RAW',label='2016B',events=100000,location='STD', ls=Run2016B)}
steps['RunMET2016B']={'INPUT':InputInfo(dataSet='/MET/Run2016B-v2/RAW',label='2016B',events=100000,location='STD', ls=Run2016B)}
steps['RunMuonEG2016B']={'INPUT':InputInfo(dataSet='/MuonEG/Run2016B-v2/RAW',label='2016B',events=100000,location='STD', ls=Run2016B)}
steps['RunDoubleEGPrpt2016B']={'INPUT':InputInfo(dataSet='/DoubleEG/Run2016B-ZElectron-PromptReco-v2/RAW-RECO',label='2016B',events=100000,location='STD', ls=Run2016B)}
steps['RunSingleMuPrpt2016B']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2016B-ZMu-PromptReco-v2/RAW-RECO',label='2016B',events=100000,location='STD', ls=Run2016B)}
steps['RunSingleEl2016B']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2016B-v2/RAW',label='2016B',events=100000,location='STD', ls={274199: [[1, 120]]})}
steps['RunSingleMu2016B']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2016B-v2/RAW',label='2016B',events=100000,location='STD', ls={274199: [[1, 120]]})}
steps['RunSinglePh2016B']={'INPUT':InputInfo(dataSet='/SinglePhoton/Run2016B-v2/RAW',label='2016B',events=100000,location='STD', ls=Run2016B)}
steps['RunZeroBias2016B']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2016B-v2/RAW',label='2016B',events=100000,location='STD', ls=Run2016B)}
steps['RunMuOnia2016B']={'INPUT':InputInfo(dataSet='/MuOnia/Run2016B-v2/RAW',label='2016B',events=100000,location='STD', ls=Run2016B)}
steps['RunNoBPTX2016B']={'INPUT':InputInfo(dataSet='/NoBPTX/Run2016B-v2/RAW',label='2016B',events=100000,location='STD', ls=Run2016B)}
# reminiAOD selection, mainly for PPS testing
steps['RunJetHT2016B_reminiaodUL']={'INPUT':InputInfo(dataSet='/JetHT/Run2016B-21Feb2020_ver2_UL2016_HIPM-v1/AOD',label='2016BrmaodUL',events=100000,location='STD', ls=Run2016B)}

#### run2 2016C ####
Run2016C={276092: [[115, 149]]}
steps['RunHLTPhy2016C']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2016C-v2/RAW',label='2016C',events=100000,location='STD', ls=Run2016C)}
steps['RunDoubleEG2016C']={'INPUT':InputInfo(dataSet='/DoubleEG/Run2016C-v2/RAW',label='2016C',events=100000,location='STD', ls=Run2016C)}
steps['RunDoubleMuon2016C']={'INPUT':InputInfo(dataSet='/DoubleMuon/Run2016C-v2/RAW',label='2016C',events=100000,location='STD', ls=Run2016C)}
steps['RunJetHT2016C']={'INPUT':InputInfo(dataSet='/JetHT/Run2016C-v2/RAW',label='2016C',events=100000,location='STD', ls=Run2016C)}
steps['RunMET2016C']={'INPUT':InputInfo(dataSet='/MET/Run2016C-v2/RAW',label='2016C',events=100000,location='STD', ls=Run2016C)}
steps['RunMuonEG2016C']={'INPUT':InputInfo(dataSet='/MuonEG/Run2016C-v2/RAW',label='2016C',events=100000,location='STD', ls=Run2016C)}
steps['RunSingleEl2016C']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2016C-v2/RAW',label='2016C',events=100000,location='STD', ls=Run2016C)}
steps['RunSingleMu2016C']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2016C-v2/RAW',label='2016C',events=100000,location='STD', ls=Run2016C)}
steps['RunSinglePh2016C']={'INPUT':InputInfo(dataSet='/SinglePhoton/Run2016C-v2/RAW',label='2016C',events=100000,location='STD', ls=Run2016C)}
steps['RunZeroBias2016C']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2016C-v2/RAW',label='2016C',events=100000,location='STD', ls=Run2016C)}
steps['RunMuOnia2016C']={'INPUT':InputInfo(dataSet='/MuOnia/Run2016C-v2/RAW',label='2016C',events=100000,location='STD', ls=Run2016C)}

#### run2 2016D ####
Run2016D={276807: [[66, 100]]}
steps['RunHLTPhy2016D']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2016D-v2/RAW',label='2016D',events=100000,location='STD', ls=Run2016D)}
steps['RunDoubleEG2016D']={'INPUT':InputInfo(dataSet='/DoubleEG/Run2016D-v2/RAW',label='2016D',events=100000,location='STD', ls=Run2016D)}
steps['RunDoubleMuon2016D']={'INPUT':InputInfo(dataSet='/DoubleMuon/Run2016D-v2/RAW',label='2016D',events=100000,location='STD', ls=Run2016D)}
steps['RunJetHT2016D']={'INPUT':InputInfo(dataSet='/JetHT/Run2016D-v2/RAW',label='2016D',events=100000,location='STD', ls=Run2016D)}
steps['RunMET2016D']={'INPUT':InputInfo(dataSet='/MET/Run2016D-v2/RAW',label='2016D',events=100000,location='STD', ls=Run2016D)}
steps['RunMuonEG2016D']={'INPUT':InputInfo(dataSet='/MuonEG/Run2016D-v2/RAW',label='2016D',events=100000,location='STD', ls=Run2016D)}
steps['RunSingleEl2016D']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2016D-v2/RAW',label='2016D',events=100000,location='STD', ls=Run2016D)}
steps['RunSingleMu2016D']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2016D-v2/RAW',label='2016D',events=100000,location='STD', ls=Run2016D)}
steps['RunSinglePh2016D']={'INPUT':InputInfo(dataSet='/SinglePhoton/Run2016D-v2/RAW',label='2016D',events=100000,location='STD', ls=Run2016D)}
steps['RunZeroBias2016D']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2016D-v2/RAW',label='2016D',events=100000,location='STD', ls=Run2016D)}
steps['RunMuOnia2016D']={'INPUT':InputInfo(dataSet='/MuOnia/Run2016D-v2/RAW',label='2016D',events=100000,location='STD', ls=Run2016D)}

#### run2 2016E ####
Run2016E={277069: [[81, 120]]}
steps['RunHLTPhy2016E']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2016E-v2/RAW',label='2016E',events=100000,location='STD', ls=Run2016E)}
steps['RunDoubleEG2016E']={'INPUT':InputInfo(dataSet='/DoubleEG/Run2016E-v2/RAW',label='2016E',events=100000,location='STD', ls=Run2016E)}
steps['RunDoubleMuon2016E']={'INPUT':InputInfo(dataSet='/DoubleMuon/Run2016E-v2/RAW',label='2016E',events=100000,location='STD', ls=Run2016E)}
steps['RunJetHT2016E']={'INPUT':InputInfo(dataSet='/JetHT/Run2016E-v2/RAW',label='2016E',events=100000,location='STD', ls=Run2016E)}
steps['RunMET2016E']={'INPUT':InputInfo(dataSet='/MET/Run2016E-v2/RAW',label='2016E',events=100000,location='STD', ls=Run2016E)}
steps['RunMuonEG2016E']={'INPUT':InputInfo(dataSet='/MuonEG/Run2016E-v2/RAW',label='2016E',events=100000,location='STD', ls=Run2016E)}
steps['RunSingleEl2016E']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2016E-v2/RAW',label='2016E',events=100000,location='STD', ls=Run2016E)}
steps['RunSingleMu2016E']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2016E-v2/RAW',label='2016E',events=100000,location='STD', ls=Run2016E)}
steps['RunSinglePh2016E']={'INPUT':InputInfo(dataSet='/SinglePhoton/Run2016E-v2/RAW',label='2016E',events=100000,location='STD', ls=Run2016E)}
steps['RunZeroBias2016E']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2016E-v2/RAW',label='2016E',events=100000,location='STD', ls=Run2016E)}
steps['RunMuOnia2016E']={'INPUT':InputInfo(dataSet='/MuOnia/Run2016E-v2/RAW',label='2016E',events=100000,location='STD', ls=Run2016E)}
steps['RunJetHT2016E_reminiaod']={'INPUT':InputInfo(dataSet='/JetHT/Run2016E-18Apr2017-v1/AOD',label='2016Ermaod',events=100000,location='STD', ls=Run2016E)}
steps['RunJetHT2016E_reminiaodUL']={'INPUT':InputInfo(dataSet='/JetHT/Run2016E-21Feb2020_UL2016_HIPM-v1/AOD',label='2016ErmaodUL',events=100000,location='STD', ls=Run2016E)}

#### run2 2016H ####
Run2016H={283877: [[1, 45]]}
steps['RunHLTPhy2016H']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2016H-v1/RAW',label='2016H',events=100000,location='STD', ls=Run2016H)}
steps['RunDoubleEG2016H']={'INPUT':InputInfo(dataSet='/DoubleEG/Run2016H-v1/RAW',label='2016H',events=100000,location='STD', ls=Run2016H)}
steps['RunDoubleMuon2016H']={'INPUT':InputInfo(dataSet='/DoubleMuon/Run2016H-v1/RAW',label='2016H',events=100000,location='STD', ls=Run2016H)}
steps['RunJetHT2016H']={'INPUT':InputInfo(dataSet='/JetHT/Run2016H-v1/RAW',label='2016H',events=100000,location='STD', ls=Run2016H)}
steps['RunMET2016H']={'INPUT':InputInfo(dataSet='/MET/Run2016H-v1/RAW',label='2016H',events=100000,location='STD', ls=Run2016H)}
steps['RunMuonEG2016H']={'INPUT':InputInfo(dataSet='/MuonEG/Run2016H-v1/RAW',label='2016H',events=100000,location='STD', ls=Run2016H)}
steps['RunSingleEl2016H']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2016H-v1/RAW',label='2016H',events=100000,location='STD', ls=Run2016H)}
steps['RunSingleMu2016H']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2016H-v1/RAW',label='2016H',events=100000,location='STD', ls=Run2016H)}
steps['RunSinglePh2016H']={'INPUT':InputInfo(dataSet='/SinglePhoton/Run2016H-v1/RAW',label='2016H',events=100000,location='STD', ls=Run2016H)}
steps['RunZeroBias2016H']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2016H-v1/RAW',label='2016H',events=100000,location='STD', ls=Run2016H)}
steps['RunMuOnia2016H']={'INPUT':InputInfo(dataSet='/MuOnia/Run2016H-v1/RAW',label='2016H',events=100000,location='STD', ls=Run2016H)}
steps['RunJetHT2016H_reminiaod']={'INPUT':InputInfo(dataSet='/JetHT/Run2016H-18Apr2017-v1/AOD',label='2016Hrmaod',events=100000,location='STD', ls=Run2016H)}
steps['RunJetHT2016H_reminiaodUL']={'INPUT':InputInfo(dataSet='/JetHT/Run2016H-21Feb2020_UL2016-v1/AOD',label='2016HrmaodUL',events=100000,location='STD', ls=Run2016H)}

#### run2 2017B ####
Run2017BlowPU={297227: [[1, 45]]}
Run2017B={297557: [[8, 167]]}
steps['RunHLTPhy2017B']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2017B-v1/RAW',label='2017B',events=100000,location='STD', ls=Run2017B)}
steps['RunDoubleEG2017B']={'INPUT':InputInfo(dataSet='/DoubleEG/Run2017B-v1/RAW',label='2017B',events=100000,location='STD', ls=Run2017B)}
steps['RunDoubleMuon2017B']={'INPUT':InputInfo(dataSet='/DoubleMuon/Run2017B-v1/RAW',label='2017B',events=100000,location='STD', ls=Run2017B)}
steps['RunJetHT2017B']={'INPUT':InputInfo(dataSet='/JetHT/Run2017B-v1/RAW',label='2017B',events=100000,location='STD', ls=Run2017B)}
steps['RunMET2017B']={'INPUT':InputInfo(dataSet='/MET/Run2017B-v1/RAW',label='2017B',events=100000,location='STD', ls=Run2017B)}
steps['RunMuonEG2017B']={'INPUT':InputInfo(dataSet='/MuonEG/Run2017B-v1/RAW',label='2017B',events=100000,location='STD', ls=Run2017B)}
steps['RunSingleEl2017B']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2017B-v1/RAW',label='2017B',events=100000,location='STD', ls=Run2017B)}
steps['RunSingleMu2017B']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2017B-v1/RAW',label='2017B',events=100000,location='STD', ls=Run2017B)}
steps['RunSinglePh2017B']={'INPUT':InputInfo(dataSet='/SinglePhoton/Run2017B-v1/RAW',label='2017B',events=100000,location='STD', ls=Run2017B)}
steps['RunZeroBias2017B']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2017B-v1/RAW',label='2017B',events=100000,location='STD', ls=Run2017B)}
steps['RunMuOnia2017B']={'INPUT':InputInfo(dataSet='/MuOnia/Run2017B-v1/RAW',label='2017B',events=100000,location='STD', ls=Run2017B)}
steps['RunCharmonium2017B']={'INPUT':InputInfo(dataSet='/Charmonium/Run2017B-v1/RAW',label='2017B',events=100000,location='STD', ls=Run2017B)}
steps['RunNoBPTX2017B']={'INPUT':InputInfo(dataSet='/NoBPTX/Run2017B-v1/RAW',label='2017B',events=100000,location='STD', ls=Run2017B)}
steps['RunHLTPhy2017B_AOD']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2017B-PromptReco-v1/AOD',label='2017Baod',events=100000,location='STD', ls=Run2017BlowPU)}
steps['RunHLTPhy2017B_AODextra']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2017B-PromptReco-v1/AOD',label='2017Baodex',events=100000,location='STD', ls=Run2017BlowPU)}
steps['RunHLTPhy2017B_RAWAOD']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2017B-PromptReco-v1/AOD',dataSetParent='/HLTPhysics/Run2017B-v1/RAW',label='2017Brawaod',events=100000,location='STD', ls=Run2017B)}

#### run2 2017C ####
Run2017C={301998: [[1, 150]]}
steps['RunHLTPhy2017C']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2017C-v1/RAW',label='2017C',events=100000,location='STD', ls=Run2017C)}
steps['RunDoubleEG2017C']={'INPUT':InputInfo(dataSet='/DoubleEG/Run2017C-v1/RAW',label='2017C',events=100000,location='STD', ls=Run2017C)}
steps['RunDoubleMuon2017C']={'INPUT':InputInfo(dataSet='/DoubleMuon/Run2017C-v1/RAW',label='2017C',events=100000,location='STD', ls=Run2017C)}
steps['RunJetHT2017C']={'INPUT':InputInfo(dataSet='/JetHT/Run2017C-v1/RAW',label='2017C',events=100000,location='STD', ls=Run2017C)}
steps['RunDisplacedJet2017C']={'INPUT':InputInfo(dataSet='/DisplacedJet/Run2017C-v1/RAW',label='2017C',events=100000,location='STD', ls=Run2017C)}
steps['RunMET2017C']={'INPUT':InputInfo(dataSet='/MET/Run2017C-v1/RAW',label='2017C',events=100000,location='STD', ls=Run2017C)}
steps['RunMuonEG2017C']={'INPUT':InputInfo(dataSet='/MuonEG/Run2017C-v1/RAW',label='2017C',events=100000,location='STD', ls=Run2017C)}
steps['RunSingleEl2017C']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2017C-v1/RAW',label='2017C',events=100000,location='STD', ls=Run2017C)}
steps['RunSingleMu2017C']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2017C-v1/RAW',label='2017C',events=100000,location='STD', ls=Run2017C)}
steps['RunSinglePh2017C']={'INPUT':InputInfo(dataSet='/SinglePhoton/Run2017C-v1/RAW',label='2017C',events=100000,location='STD', ls=Run2017C)}
steps['RunZeroBias2017C']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2017C-v1/RAW',label='2017C',events=100000,location='STD', ls=Run2017C)}
steps['RunMuOnia2017C']={'INPUT':InputInfo(dataSet='/MuOnia/Run2017C-v1/RAW',label='2017C',events=100000,location='STD', ls=Run2017C)}
steps['RunCharmonium2017C']={'INPUT':InputInfo(dataSet='/Charmonium/Run2017C-v1/RAW',label='2017C',events=100000,location='STD', ls=Run2017C)}
steps['RunNoBPTX2017C']={'INPUT':InputInfo(dataSet='/NoBPTX/Run2017C-v1/RAW',label='2017C',events=100000,location='STD', ls=Run2017C)}



#### run2 2017D ####
Run2017D={302663: [[1, 100]]}  #AVGPU 36
steps['RunHLTPhy2017D']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2017D-v1/RAW',label='2017D',events=100000,location='STD', ls=Run2017D)}
steps['RunDoubleEG2017D']={'INPUT':InputInfo(dataSet='/DoubleEG/Run2017D-v1/RAW',label='2017D',events=100000,location='STD', ls=Run2017D)}
steps['RunDoubleMuon2017D']={'INPUT':InputInfo(dataSet='/DoubleMuon/Run2017D-v1/RAW',label='2017D',events=100000,location='STD', ls=Run2017D)}
steps['RunJetHT2017D']={'INPUT':InputInfo(dataSet='/JetHT/Run2017D-v1/RAW',label='2017D',events=100000,location='STD', ls=Run2017D)}
steps['RunDisplacedJet2017D']={'INPUT':InputInfo(dataSet='/DisplacedJet/Run2017D-v1/RAW',label='2017D',events=100000,location='STD', ls=Run2017D)}
steps['RunMET2017D']={'INPUT':InputInfo(dataSet='/MET/Run2017D-v1/RAW',label='2017D',events=100000,location='STD', ls=Run2017D)}
steps['RunMuonEG2017D']={'INPUT':InputInfo(dataSet='/MuonEG/Run2017D-v1/RAW',label='2017D',events=100000,location='STD', ls=Run2017D)}
steps['RunSingleEl2017D']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2017D-v1/RAW',label='2017D',events=100000,location='STD', ls=Run2017D)}
steps['RunSingleMu2017D']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2017D-v1/RAW',label='2017D',events=100000,location='STD', ls=Run2017D)}
steps['RunSinglePh2017D']={'INPUT':InputInfo(dataSet='/SinglePhoton/Run2017D-v1/RAW',label='2017D',events=100000,location='STD', ls=Run2017D)}
steps['RunZeroBias2017D']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2017D-v1/RAW',label='2017D',events=100000,location='STD', ls=Run2017D)}
steps['RunMuOnia2017D']={'INPUT':InputInfo(dataSet='/MuOnia/Run2017D-v1/RAW',label='2017D',events=100000,location='STD', ls=Run2017D)}
steps['RunCharmonium2017D']={'INPUT':InputInfo(dataSet='/Charmonium/Run2017D-v1/RAW',label='2017D',events=100000,location='STD', ls=Run2017D)}
steps['RunNoBPTX2017D']={'INPUT':InputInfo(dataSet='/NoBPTX/Run2017D-v1/RAW',label='2017D',events=100000,location='STD', ls=Run2017D)}


#### run2 2017E ####
Run2017E={304125: [[1, 100]]} #AVGPU 46
steps['RunHLTPhy2017E']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2017E-v1/RAW',label='2017E',events=100000,location='STD', ls=Run2017E)}
steps['RunDoubleEG2017E']={'INPUT':InputInfo(dataSet='/DoubleEG/Run2017E-v1/RAW',label='2017E',events=100000,location='STD', ls=Run2017E)}
steps['RunDoubleMuon2017E']={'INPUT':InputInfo(dataSet='/DoubleMuon/Run2017E-v1/RAW',label='2017E',events=100000,location='STD', ls=Run2017E)}
steps['RunJetHT2017E']={'INPUT':InputInfo(dataSet='/JetHT/Run2017E-v1/RAW',label='2017E',events=100000,location='STD', ls=Run2017E)}
steps['RunDisplacedJet2017E']={'INPUT':InputInfo(dataSet='/DisplacedJet/Run2017E-v1/RAW',label='2017E',events=100000,location='STD', ls=Run2017E)}
steps['RunMET2017E']={'INPUT':InputInfo(dataSet='/MET/Run2017E-v1/RAW',label='2017E',events=100000,location='STD', ls=Run2017E)}
steps['RunMuonEG2017E']={'INPUT':InputInfo(dataSet='/MuonEG/Run2017E-v1/RAW',label='2017E',events=100000,location='STD', ls=Run2017E)}
steps['RunSingleEl2017E']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2017E-v1/RAW',label='2017E',events=100000,location='STD', ls=Run2017E)}
steps['RunSingleMu2017E']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2017E-v1/RAW',label='2017E',events=100000,location='STD', ls=Run2017E)}
steps['RunSinglePh2017E']={'INPUT':InputInfo(dataSet='/SinglePhoton/Run2017E-v1/RAW',label='2017E',events=100000,location='STD', ls=Run2017E)}
steps['RunZeroBias2017E']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2017E-v1/RAW',label='2017E',events=100000,location='STD', ls=Run2017E)}
steps['RunMuOnia2017E']={'INPUT':InputInfo(dataSet='/MuOnia/Run2017E-v1/RAW',label='2017E',events=100000,location='STD', ls=Run2017E)}
steps['RunCharmonium2017E']={'INPUT':InputInfo(dataSet='/Charmonium/Run2017E-v1/RAW',label='2017E',events=100000,location='STD', ls=Run2017E)}
steps['RunNoBPTX2017E']={'INPUT':InputInfo(dataSet='/NoBPTX/Run2017E-v1/RAW',label='2017E',events=100000,location='STD', ls=Run2017E)}

#### run2 2017F ####
Run2017F={305064: [[2, 101]]} #AVGPU 51
steps['RunHLTPhy2017F']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2017F-v1/RAW',label='2017F',events=100000,location='STD', ls=Run2017F)}
steps['RunDoubleEG2017F']={'INPUT':InputInfo(dataSet='/DoubleEG/Run2017F-v1/RAW',label='2017F',events=100000,location='STD', ls=Run2017F)}
steps['RunDoubleMuon2017F']={'INPUT':InputInfo(dataSet='/DoubleMuon/Run2017F-v1/RAW',label='2017F',events=100000,location='STD', ls=Run2017F)}
steps['RunJetHT2017F']={'INPUT':InputInfo(dataSet='/JetHT/Run2017F-v1/RAW',label='2017F',events=100000,location='STD', ls=Run2017F)}
steps['RunDisplacedJet2017F']={'INPUT':InputInfo(dataSet='/DisplacedJet/Run2017F-v1/RAW',label='2017F',events=100000,location='STD', ls=Run2017F)}
steps['RunMET2017F']={'INPUT':InputInfo(dataSet='/MET/Run2017F-v1/RAW',label='2017F',events=100000,location='STD', ls=Run2017F)}
steps['RunMuonEG2017F']={'INPUT':InputInfo(dataSet='/MuonEG/Run2017F-v1/RAW',label='2017F',events=100000,location='STD', ls=Run2017F)}
steps['RunSingleEl2017F']={'INPUT':InputInfo(dataSet='/SingleElectron/Run2017F-v1/RAW',label='2017F',events=100000,location='STD', ls=Run2017F)}
steps['RunSingleMu2017F']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2017F-v1/RAW',label='2017F',events=100000,location='STD', ls=Run2017F)}
steps['RunSinglePh2017F']={'INPUT':InputInfo(dataSet='/SinglePhoton/Run2017F-v1/RAW',label='2017F',events=100000,location='STD', ls=Run2017F)}
steps['RunZeroBias2017F']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2017F-v1/RAW',label='2017F',events=100000,location='STD', ls=Run2017F)}
steps['RunMuOnia2017F']={'INPUT':InputInfo(dataSet='/MuOnia/Run2017F-v1/RAW',label='2017F',events=100000,location='STD', ls=Run2017F)}
steps['RunCharmonium2017F']={'INPUT':InputInfo(dataSet='/Charmonium/Run2017F-v1/RAW',label='2017F',events=100000,location='STD', ls=Run2017F)}
steps['RunNoBPTX2017F']={'INPUT':InputInfo(dataSet='/NoBPTX/Run2017F-v1/RAW',label='2017F',events=100000,location='STD', ls=Run2017F)}
steps['RunExpressPhy2017F']={'INPUT':InputInfo(dataSet='/ExpressPhysics/Run2017F-Express-v1/FEVT',label='2017F',events=100000,location='STD', ls=Run2017F)}

steps['RunJetHT2017F_reminiaod']={'INPUT':InputInfo(dataSet='/JetHT/Run2017F-17Nov2017-v1/AOD',label='2017Frmaod',events=100000,location='STD', ls=Run2017F)}
steps['RunJetHT2017F_reminiaodUL']={'INPUT':InputInfo(dataSet='/JetHT/Run2017F-09Aug2019_UL2017-v1/AOD',label='2017FrmaodUL',events=100000,location='STD', ls=Run2017F)}


#### run2 2018A ####
Run2018A={315489: [[1, 100]]} #AVGPU 35
steps['RunHLTPhy2018A']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2018A-v1/RAW',label='2018A',events=100000,location='STD', ls=Run2018A)}
steps['RunEGamma2018A']={'INPUT':InputInfo(dataSet='/EGamma/Run2018A-v1/RAW',label='2018A',events=100000,location='STD', ls=Run2018A)}
steps['RunDoubleMuon2018A']={'INPUT':InputInfo(dataSet='/DoubleMuon/Run2018A-v1/RAW',label='2018A',events=100000,location='STD', ls=Run2018A)}
steps['RunJetHT2018A']={'INPUT':InputInfo(dataSet='/JetHT/Run2018A-v1/RAW',label='2018A',events=100000,location='STD', ls=Run2018A)}
steps['RunMET2018A']={'INPUT':InputInfo(dataSet='/MET/Run2018A-v1/RAW',label='2018A',events=100000,location='STD', ls=Run2018A)}
steps['RunMuonEG2018A']={'INPUT':InputInfo(dataSet='/MuonEG/Run2018A-v1/RAW',label='2018A',events=100000,location='STD', ls=Run2018A)}
steps['RunSingleMu2018A']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2018A-v1/RAW',label='2018A',events=100000,location='STD', ls=Run2018A)}
steps['RunZeroBias2018A']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2018A-v1/RAW',label='2018A',events=100000,location='STD', ls=Run2018A)}
steps['RunMuOnia2018A']={'INPUT':InputInfo(dataSet='/MuOnia/Run2018A-v1/RAW',label='2018A',events=100000,location='STD', ls=Run2018A)}
steps['RunNoBPTX2018A']={'INPUT':InputInfo(dataSet='/NoBPTX/Run2018A-v1/RAW',label='2018A',events=100000,location='STD', ls=Run2018A)}
steps['RunDisplacedJet2018A']={'INPUT':InputInfo(dataSet='/DisplacedJet/Run2018A-v1/RAW',label='2018A',events=100000,location='STD', ls=Run2018A)}
steps['RunCharmonium2018A']={'INPUT':InputInfo(dataSet='/Charmonium/Run2018A-v1/RAW',label='2018A',events=100000,location='STD', ls=Run2018A)}
#### for 90 m beta* Totem run relvals ####
RunhBStarTk={314890: [[500, 700]]} #for central tracking system
steps['RunZeroBias_hBStarTk']={'INPUT':InputInfo(dataSet='/ZeroBias/Commissioning2018-v1/RAW',label='zbhBSTk',events=100000,location='STD', ls=RunhBStarTk)}
RunhBStarRP={314276: [[1, 200]]} #for Roman Pot system
steps['RunZeroBias1_hBStarRP']={'INPUT':InputInfo(dataSet='/ZeroBias1/Commissioning2018-v1/RAW',label='zbhBSRP',events=100000,location='STD', ls=RunhBStarRP)}


#### run2 2018B ####
Run2018B={317435: [[1, 100]]}
Run2018B_parkingBPH={317661: [[301, 400]]}
steps['RunHLTPhy2018B']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2018B-v1/RAW',label='2018B',events=100000,location='STD', ls=Run2018B)}
steps['RunEGamma2018B']={'INPUT':InputInfo(dataSet='/EGamma/Run2018B-v1/RAW',label='2018B',events=100000,location='STD', ls=Run2018B)}
steps['RunDoubleMuon2018B']={'INPUT':InputInfo(dataSet='/DoubleMuon/Run2018B-v1/RAW',label='2018B',events=100000,location='STD', ls=Run2018B)}
steps['RunJetHT2018B']={'INPUT':InputInfo(dataSet='/JetHT/Run2018B-v1/RAW',label='2018B',events=100000,location='STD', ls=Run2018B)}
steps['RunMET2018B']={'INPUT':InputInfo(dataSet='/MET/Run2018B-v1/RAW',label='2018B',events=100000,location='STD', ls=Run2018B)}
steps['RunMuonEG2018B']={'INPUT':InputInfo(dataSet='/MuonEG/Run2018B-v1/RAW',label='2018B',events=100000,location='STD', ls=Run2018B)}
steps['RunSingleMu2018B']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2018B-v1/RAW',label='2018B',events=100000,location='STD', ls=Run2018B)}
steps['RunZeroBias2018B']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2018B-v1/RAW',label='2018B',events=100000,location='STD', ls=Run2018B)}
steps['RunMuOnia2018B']={'INPUT':InputInfo(dataSet='/MuOnia/Run2018B-v1/RAW',label='2018B',events=100000,location='STD', ls=Run2018B)}
steps['RunNoBPTX2018B']={'INPUT':InputInfo(dataSet='/NoBPTX/Run2018B-v1/RAW',label='2018B',events=100000,location='STD', ls=Run2018B)}
steps['RunDisplacedJet2018B']={'INPUT':InputInfo(dataSet='/DisplacedJet/Run2018B-v1/RAW',label='2018B',events=100000,location='STD', ls=Run2018B)}
steps['RunCharmonium2018B']={'INPUT':InputInfo(dataSet='/Charmonium/Run2018B-v1/RAW',label='2018B',events=100000,location='STD', ls=Run2018B)}
steps['RunParkingBPH2018B']={'INPUT':InputInfo(dataSet='/ParkingBPH5/Run2018B-v1/RAW',label='2018B',events=100000,location='STD', ls=Run2018B_parkingBPH)}
steps['RunJetHT2018BHEfail']={'INPUT':InputInfo(dataSet='/JetHT/Run2018B-v1/RAW',label='HEfail',events=100000,location='STD', ls=Run2018B)}

#### run2 2018C ####
Run2018C={319450: [[1, 100]]}
steps['RunHLTPhy2018C']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2018C-v1/RAW',label='2018C',events=100000,location='STD', ls=Run2018C)}
steps['RunEGamma2018C']={'INPUT':InputInfo(dataSet='/EGamma/Run2018C-v1/RAW',label='2018C',events=100000,location='STD', ls=Run2018C)}
steps['RunDoubleMuon2018C']={'INPUT':InputInfo(dataSet='/DoubleMuon/Run2018C-v1/RAW',label='2018C',events=100000,location='STD', ls=Run2018C)}
steps['RunJetHT2018C']={'INPUT':InputInfo(dataSet='/JetHT/Run2018C-v1/RAW',label='2018C',events=100000,location='STD', ls=Run2018C)}
steps['RunMET2018C']={'INPUT':InputInfo(dataSet='/MET/Run2018C-v1/RAW',label='2018C',events=100000,location='STD', ls=Run2018C)}
steps['RunMuonEG2018C']={'INPUT':InputInfo(dataSet='/MuonEG/Run2018C-v1/RAW',label='2018C',events=100000,location='STD', ls=Run2018C)}
steps['RunSingleMu2018C']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2018C-v1/RAW',label='2018C',events=100000,location='STD', ls=Run2018C)}
steps['RunZeroBias2018C']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2018C-v1/RAW',label='2018C',events=100000,location='STD', ls=Run2018C)}
steps['RunMuOnia2018C']={'INPUT':InputInfo(dataSet='/MuOnia/Run2018C-v1/RAW',label='2018C',events=100000,location='STD', ls=Run2018C)}
steps['RunNoBPTX2018C']={'INPUT':InputInfo(dataSet='/NoBPTX/Run2018C-v1/RAW',label='2018C',events=100000,location='STD', ls=Run2018C)}
steps['RunDisplacedJet2018C']={'INPUT':InputInfo(dataSet='/DisplacedJet/Run2018C-v1/RAW',label='2018C',events=100000,location='STD', ls=Run2018C)}
steps['RunCharmonium2018C']={'INPUT':InputInfo(dataSet='/Charmonium/Run2018C-v1/RAW',label='2018C',events=100000,location='STD', ls=Run2018C)}


#### run2 2018D ####
Run2018D={320822: [[1, 100]]}
steps['RunHLTPhy2018D']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2018D-v1/RAW',label='2018D',events=100000,location='STD', ls=Run2018D)}
steps['RunEGamma2018D']={'INPUT':InputInfo(dataSet='/EGamma/Run2018D-v1/RAW',label='2018D',events=100000,location='STD', ls=Run2018D)}
steps['RunDoubleMuon2018D']={'INPUT':InputInfo(dataSet='/DoubleMuon/Run2018D-v1/RAW',label='2018D',events=100000,location='STD', ls=Run2018D)}
steps['RunJetHT2018D']={'INPUT':InputInfo(dataSet='/JetHT/Run2018D-v1/RAW',label='2018D',events=100000,location='STD', ls=Run2018D)}
steps['RunMET2018D']={'INPUT':InputInfo(dataSet='/MET/Run2018D-v1/RAW',label='2018D',events=100000,location='STD', ls=Run2018D)}
steps['RunMuonEG2018D']={'INPUT':InputInfo(dataSet='/MuonEG/Run2018D-v1/RAW',label='2018D',events=100000,location='STD', ls=Run2018D)}
steps['RunSingleMu2018D']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2018D-v1/RAW',label='2018D',events=100000,location='STD', ls=Run2018D)}
steps['RunZeroBias2018D']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2018D-v1/RAW',label='2018D',events=100000,location='STD', ls=Run2018D)}
steps['RunMuOnia2018D']={'INPUT':InputInfo(dataSet='/MuOnia/Run2018D-v1/RAW',label='2018D',events=100000,location='STD', ls=Run2018D)}
steps['RunNoBPTX2018D']={'INPUT':InputInfo(dataSet='/NoBPTX/Run2018D-v1/RAW',label='2018D',events=100000,location='STD', ls=Run2018D)}
steps['RunDisplacedJet2018D']={'INPUT':InputInfo(dataSet='/DisplacedJet/Run2018D-v1/RAW',label='2018D',events=100000,location='STD', ls=Run2018D)}
steps['RunCharmonium2018D']={'INPUT':InputInfo(dataSet='/Charmonium/Run2018D-v1/RAW',label='2018D',events=100000,location='STD', ls=Run2018D)}

#### PPS run3 2022B
steps['RunRawPPS2022B']={'INPUT':InputInfo(dataSet='/AlCaPPS/Run2022B-v1/RAW',label='2022B',events=100000,location='STD', ls={355207: [[1, 100]]})}
steps['RunRawPPS2022A']={'INPUT':InputInfo(dataSet='/AlCaPPS/Run2022A-v1/RAW',label='2022A',events=100000,location='STD', ls={354332: [[1, 100]]})}

# UL AOD
steps['RunJetHT2018D_reminiaodUL']={'INPUT':InputInfo(dataSet='/JetHT/Run2018D-12Nov2019_UL2018-v4/AOD',label='2018DrmaodUL',events=100000,location='STD', ls=Run2018D)}

#### run3 ####
Run2022A={353015: [[1, 100]]}
steps['RunMinimumBias2022A']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2022A-v1/RAW',label='2022A',events=100000,location='STD', ls=Run2022A)}
steps['RunSingleMuon2022A']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2022A-v1/RAW',label='2022A',events=100000,location='STD', ls=Run2022A)}
steps['RunZeroBias2022A']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2022A-v1/RAW',label='2022A',events=100000,location='STD', ls=Run2022A)}
steps['RunBTagMu2022A']={'INPUT':InputInfo(dataSet='/BTagMu/Run2022A-v1/RAW',label='2022A',events=100000,location='STD', ls=Run2022A)}
steps['RunNoBPTX2022A']={'INPUT':InputInfo(dataSet='/NoBPTX/Run2022A-v1/RAW',label='2022A',events=100000,location='STD', ls=Run2022A)}
steps['RunHcalNZS2022A']={'INPUT':InputInfo(dataSet='/HcalNZS/Run2022A-v1/RAW',label='2022A',events=100000,location='STD', ls=Run2022A)}
steps['RunHLTPhysics2022A']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2022A-v1/RAW',label='2022A',events=100000,location='STD', ls=Run2022A)}
steps['RunCommissioning2022A']={'INPUT':InputInfo(dataSet='/Commissioning/Run2022A-v1/RAW',label='2022A',events=100000,location='STD', ls=Run2022A)}
steps['RunJetHT2022A']={'INPUT':InputInfo(dataSet='/JetHT/Run2022A-v1/RAW',label='2022A',events=100000,location='STD', ls=Run2022A)}
steps['RunCosmics2022A']={'INPUT':InputInfo(dataSet='/Cosmics/Run2022A-v1/RAW',label='2022A',events=100000,location='STD', ls=Run2022A)}
steps['RunDisplacedJet2022A']={'INPUT':InputInfo(dataSet='/DisplacedJet/Run2022A-v1/RAW',label='2022A',events=100000,location='STD', ls=Run2022A)}
steps['RunMET2022A']={'INPUT':InputInfo(dataSet='/MET/Run2022A-v1/RAW',label='2022A',events=100000,location='STD', ls=Run2022A)}
steps['RunEGamma2022A']={'INPUT':InputInfo(dataSet='/EGamma/Run2022A-v1/RAW',label='2022A',events=100000,location='STD', ls=Run2022A)}
steps['RunTau2022A']={'INPUT':InputInfo(dataSet='/Tau/Run2022A-v1/RAW',label='2022A',events=100000,location='STD', ls=Run2022A)}
steps['RunDoubleMuon2022A']={'INPUT':InputInfo(dataSet='/DoubleMuon/Run2022A-v1/RAW',label='2022A',events=100000,location='STD', ls=Run2022A)}
steps['RunMuonEG2022A']={'INPUT':InputInfo(dataSet='/MuonEG/Run2022A-v1/RAW',label='2022A',events=100000,location='STD', ls=Run2022A)}

Run2022B={355769: [[1, 106]]}
steps['RunMinimumBias2022B']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2022B-v1/RAW',label='2022B',events=100000,location='STD', ls=Run2022B)}
steps['RunSingleMuon2022B']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2022B-v1/RAW',label='2022B',events=100000,location='STD', ls=Run2022B)}
steps['RunZeroBias2022B']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2022B-v1/RAW',label='2022B',events=100000,location='STD', ls=Run2022B)}
steps['RunBTagMu2022B']={'INPUT':InputInfo(dataSet='/BTagMu/Run2022B-v1/RAW',label='2022B',events=100000,location='STD', ls=Run2022B)}
steps['RunNoBPTX2022B']={'INPUT':InputInfo(dataSet='/NoBPTX/Run2022B-v1/RAW',label='2022B',events=100000,location='STD', ls=Run2022B)}
steps['RunHcalNZS2022B']={'INPUT':InputInfo(dataSet='/HcalNZS/Run2022B-v1/RAW',label='2022B',events=100000,location='STD', ls=Run2022B)}
steps['RunHLTPhysics2022B']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2022B-v1/RAW',label='2022B',events=100000,location='STD', ls=Run2022B)}
steps['RunCommissioning2022B']={'INPUT':InputInfo(dataSet='/Commissioning/Run2022B-v1/RAW',label='2022B',events=100000,location='STD', ls=Run2022B)}
steps['RunJetHT2022B']={'INPUT':InputInfo(dataSet='/JetHT/Run2022B-v1/RAW',label='2022B',events=100000,location='STD', ls=Run2022B)}
steps['RunCosmics2022B']={'INPUT':InputInfo(dataSet='/Cosmics/Run2022B-v1/RAW',label='2022B',events=100000,location='STD', ls=Run2022B)}
steps['RunDisplacedJet2022B']={'INPUT':InputInfo(dataSet='/DisplacedJet/Run2022B-v1/RAW',label='2022B',events=100000,location='STD', ls=Run2022B)}
steps['RunMET2022B']={'INPUT':InputInfo(dataSet='/MET/Run2022B-v1/RAW',label='2022B',events=100000,location='STD', ls=Run2022B)}
steps['RunEGamma2022B']={'INPUT':InputInfo(dataSet='/EGamma/Run2022B-v1/RAW',label='2022B',events=100000,location='STD', ls=Run2022B)}
steps['RunTau2022B']={'INPUT':InputInfo(dataSet='/Tau/Run2022B-v1/RAW',label='2022B',events=100000,location='STD', ls=Run2022B)}
steps['RunDoubleMuon2022B']={'INPUT':InputInfo(dataSet='/DoubleMuon/Run2022B-v1/RAW',label='2022B',events=100000,location='STD', ls=Run2022B)}
steps['RunMuonEG2022B']={'INPUT':InputInfo(dataSet='/MuonEG/Run2022B-v1/RAW',label='2022B',events=100000,location='STD', ls=Run2022B)}
#steps['RunParkingBPH2022B']={'INPUT':InputInfo(dataSet='/ParkingBPH/Run2022B-v1/RAW',label='2022B',events=100000,location='STD', ls=Run2022B)}

Run2022C={356381: [[1, 1193]]}
Run2022C_LS40={356381: [[1, 40]]}
Run2022C_LS75={356381: [[1, 75]]}
Run2022C_LS150={356381: [[1, 150]]}
steps['RunMinimumBias2022C']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2022C-v1/RAW',label='2022C',events=100000,location='STD', ls=Run2022C)}
steps['RunSingleMuon2022C']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2022C-v1/RAW',label='2022C',events=100000,location='STD', ls=Run2022C_LS75)}
steps['RunZeroBias2022C']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2022C-v1/RAW',label='2022C',events=100000,location='STD', ls=Run2022C_LS75)}
steps['RunBTagMu2022C']={'INPUT':InputInfo(dataSet='/BTagMu/Run2022C-v1/RAW',label='2022C',events=100000,location='STD', ls=Run2022C_LS150)}
steps['RunNoBPTX2022C']={'INPUT':InputInfo(dataSet='/NoBPTX/Run2022C-v1/RAW',label='2022C',events=100000,location='STD', ls=Run2022C)}
steps['RunHcalNZS2022C']={'INPUT':InputInfo(dataSet='/HcalNZS/Run2022C-v1/RAW',label='2022C',events=100000,location='STD', ls=Run2022C)}
steps['RunHLTPhysics2022C']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2022C-v1/RAW',label='2022C',events=100000,location='STD', ls=Run2022C)}
steps['RunCommissioning2022C']={'INPUT':InputInfo(dataSet='/Commissioning/Run2022C-v1/RAW',label='2022C',events=100000,location='STD', ls=Run2022C)}
steps['RunJetHT2022C']={'INPUT':InputInfo(dataSet='/JetHT/Run2022C-v1/RAW',label='2022C',events=100000,location='STD', ls=Run2022C_LS75)}
steps['RunCosmics2022C']={'INPUT':InputInfo(dataSet='/Cosmics/Run2022C-v1/RAW',label='2022C',events=100000,location='STD', ls=Run2022C)}
steps['RunDisplacedJet2022C']={'INPUT':InputInfo(dataSet='/DisplacedJet/Run2022C-v1/RAW',label='2022C',events=100000,location='STD', ls=Run2022C_LS150)}
steps['RunMET2022C']={'INPUT':InputInfo(dataSet='/MET/Run2022C-v1/RAW',label='2022C',events=100000,location='STD', ls=Run2022C_LS150)}
steps['RunEGamma2022C']={'INPUT':InputInfo(dataSet='/EGamma/Run2022C-v1/RAW',label='2022C',events=100000,location='STD', ls=Run2022C_LS40)}
steps['RunTau2022C']={'INPUT':InputInfo(dataSet='/Tau/Run2022C-v1/RAW',label='2022C',events=100000,location='STD', ls=Run2022C_LS150)}
steps['RunDoubleMuon2022C']={'INPUT':InputInfo(dataSet='/DoubleMuon/Run2022C-v1/RAW',label='2022C',events=100000,location='STD', ls=Run2022C_LS150)}
steps['RunMuonEG2022C']={'INPUT':InputInfo(dataSet='/MuonEG/Run2022C-v1/RAW',label='2022C',events=100000,location='STD', ls=Run2022C_LS150)}

Run2022D={357735: [[1, 366]]}
Run2022D_LS100={357735: [[1, 100]]}
Run2022D_LS25={357735: [[1, 25]]}
steps['RunMinimumBias2022D']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2022D-v1/RAW',label='2022D',events=100000,location='STD', ls=Run2022D)}
steps['RunMuon2022D']={'INPUT':InputInfo(dataSet='/Muon/Run2022D-v1/RAW',label='2022D',events=100000,location='STD', ls=Run2022D_LS25)}
steps['RunZeroBias2022D']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2022D-v1/RAW',label='2022D',events=100000,location='STD', ls=Run2022D_LS100)}
steps['RunBTagMu2022D']={'INPUT':InputInfo(dataSet='/BTagMu/Run2022D-v1/RAW',label='2022D',events=100000,location='STD', ls=Run2022D_LS100)}
steps['RunNoBPTX2022D']={'INPUT':InputInfo(dataSet='/NoBPTX/Run2022D-v1/RAW',label='2022D',events=100000,location='STD', ls=Run2022D)}
steps['RunHcalNZS2022D']={'INPUT':InputInfo(dataSet='/HcalNZS/Run2022D-v1/RAW',label='2022D',events=100000,location='STD', ls=Run2022D)}
steps['RunHLTPhysics2022D']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2022D-v1/RAW',label='2022D',events=100000,location='STD', ls=Run2022D)}
steps['RunCommissioning2022D']={'INPUT':InputInfo(dataSet='/Commissioning/Run2022D-v1/RAW',label='2022D',events=100000,location='STD', ls=Run2022D)}
steps['RunJetMET2022D']={'INPUT':InputInfo(dataSet='/JetMET/Run2022D-v1/RAW',label='2022D',events=100000,location='STD', ls=Run2022D_LS25)}
steps['RunCosmics2022D']={'INPUT':InputInfo(dataSet='/Cosmics/Run2022D-v1/RAW',label='2022D',events=100000,location='STD', ls=Run2022D)}
steps['RunDisplacedJet2022D']={'INPUT':InputInfo(dataSet='/DisplacedJet/Run2022D-v1/RAW',label='2022D',events=100000,location='STD', ls=Run2022D_LS100)}
steps['RunEGamma2022D']={'INPUT':InputInfo(dataSet='/EGamma/Run2022D-v1/RAW',label='2022D',events=100000,location='STD', ls=Run2022D_LS25)}
steps['RunTau2022D']={'INPUT':InputInfo(dataSet='/Tau/Run2022D-v1/RAW',label='2022D',events=100000,location='STD', ls=Run2022D_LS100)}
steps['RunMuonEG2022D']={'INPUT':InputInfo(dataSet='/MuonEG/Run2022D-v1/RAW',label='2022D',events=100000,location='STD', ls=Run2022D_LS100)}


Run2022E={359694: [[1, 50]]}
Run2022E_LS100={359694: [[1, 100]]}
Run2022E_LS200={359694: [[1, 200]]}
steps['RunMuon2022E']={'INPUT':InputInfo(dataSet='/Muon/Run2022E-v1/RAW',label='2022E',events=100000,location='STD', ls=Run2022E)}
steps['RunZeroBias2022E']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2022E-v1/RAW',label='2022E',events=100000,location='STD', ls=Run2022E_LS100)}
steps['RunBTagMu2022E']={'INPUT':InputInfo(dataSet='/BTagMu/Run2022E-v1/RAW',label='2022E',events=100000,location='STD', ls=Run2022E_LS100)}
steps['RunJetMET2022E']={'INPUT':InputInfo(dataSet='/JetMET/Run2022E-v1/RAW',label='2022E',events=100000,location='STD', ls=Run2022E)}
steps['RunDisplacedJet2022E']={'INPUT':InputInfo(dataSet='/DisplacedJet/Run2022E-v1/RAW',label='2022E',events=100000,location='STD', ls=Run2022E_LS200)}
steps['RunEGamma2022E']={'INPUT':InputInfo(dataSet='/EGamma/Run2022E-v1/RAW',label='2022E',events=100000,location='STD', ls=Run2022E)}
steps['RunTau2022E']={'INPUT':InputInfo(dataSet='/Tau/Run2022E-v1/RAW',label='2022E',events=100000,location='STD', ls=Run2022E_LS200)}
steps['RunMuonEG2022E']={'INPUT':InputInfo(dataSet='/MuonEG/Run2022E-v1/RAW',label='2022E',events=100000,location='STD', ls=Run2022E_LS200)}


Run2022FCosmics_LS24={360336: [[24, 24]]}
Run2022FCosmics={360333: [[1,105]]}
steps['RunCosmics2022F']={'INPUT':InputInfo(dataSet='/Cosmics/Run2022F-v1/RAW',label='2022F',events=100000,location='STD', ls=Run2022FCosmics_LS24)}
steps['RunSiPixelCalCosmics2022F']={'INPUT':InputInfo(dataSet='/Cosmics/Run2022F-SiPixelCalCosmics-PromptReco-v1/ALCARECO',label='2022F',events=100000,location='STD', ls=Run2022FCosmics)}

#### run3 ####
Run2023B={366727: [[1, 244]]}
steps['RunMuon2023B']={'INPUT':InputInfo(dataSet='/Muon0/Run2023B-v1/RAW',label='2023B',events=100000,location='STD', ls=Run2023B)}
steps['RunZeroBias2023B']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2023B-v1/RAW',label='2023B',events=100000,location='STD', ls=Run2023B)}
steps['RunBTagMu2023B']={'INPUT':InputInfo(dataSet='/BTagMu/Run2023B-v1/RAW',label='2023B',events=100000,location='STD', ls=Run2023B)}
steps['RunNoBPTX2023B']={'INPUT':InputInfo(dataSet='/NoBPTX/Run2023B-v1/RAW',label='2023B',events=100000,location='STD', ls=Run2023B)}
steps['RunHcalNZS2023B']={'INPUT':InputInfo(dataSet='/HcalNZS/Run2023B-v1/RAW',label='2023B',events=100000,location='STD', ls=Run2023B)}
steps['RunHLTPhysics2023B']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2023B-v1/RAW',label='2023B',events=100000,location='STD', ls=Run2023B)}
steps['RunCommissioning2023B']={'INPUT':InputInfo(dataSet='/Commissioning/Run2023B-v1/RAW',label='2023B',events=100000,location='STD', ls=Run2023B)}
steps['RunJetMET2023B']={'INPUT':InputInfo(dataSet='/JetMET1/Run2023B-v1/RAW',label='2023B',events=100000,location='STD', ls=Run2023B)}
steps['RunCosmics2023B']={'INPUT':InputInfo(dataSet='/Cosmics/Run2023B-v1/RAW',label='2023B',events=100000,location='STD', ls=Run2023B)}
steps['RunDisplacedJet2023B']={'INPUT':InputInfo(dataSet='/DisplacedJet/Run2023B-v1/RAW',label='2023B',events=100000,location='STD', ls=Run2023B)}
steps['RunEGamma2023B']={'INPUT':InputInfo(dataSet='/EGamma1/Run2023B-v1/RAW',label='2023B',events=100000,location='STD', ls=Run2023B)}
steps['RunTau2023B']={'INPUT':InputInfo(dataSet='/Tau/Run2023B-v1/RAW',label='2023B',events=100000,location='STD', ls=Run2023B)}
steps['RunMuonEG2023B']={'INPUT':InputInfo(dataSet='/MuonEG/Run2023B-v1/RAW',label='2023B',events=100000,location='STD', ls=Run2023B)}

Run2023C={367131: [[1, 149]]}
steps['RunMuon2023C']={'INPUT':InputInfo(dataSet='/Muon0/Run2023C-v1/RAW',label='2023C',events=100000,location='STD', ls=Run2023C)}
steps['RunZeroBias2023C']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2023C-v1/RAW',label='2023C',events=100000,location='STD', ls=Run2023C)}
steps['RunBTagMu2023C']={'INPUT':InputInfo(dataSet='/BTagMu/Run2023C-v1/RAW',label='2023C',events=100000,location='STD', ls=Run2023C)}
steps['RunNoBPTX2023C']={'INPUT':InputInfo(dataSet='/NoBPTX/Run2023C-v1/RAW',label='2023C',events=100000,location='STD', ls=Run2023C)}
steps['RunHcalNZS2023C']={'INPUT':InputInfo(dataSet='/HcalNZS/Run2023C-v1/RAW',label='2023C',events=100000,location='STD', ls=Run2023C)}
steps['RunHLTPhysics2023C']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2023C-v1/RAW',label='2023C',events=100000,location='STD', ls=Run2023C)}
steps['RunCommissioning2023C']={'INPUT':InputInfo(dataSet='/Commissioning/Run2023C-v1/RAW',label='2023C',events=100000,location='STD', ls=Run2023C)}
steps['RunJetMET2023C']={'INPUT':InputInfo(dataSet='/JetMET1/Run2023C-v1/RAW',label='2023C',events=100000,location='STD', ls=Run2023C)}
steps['RunCosmics2023C']={'INPUT':InputInfo(dataSet='/Cosmics/Run2023C-v1/RAW',label='2023C',events=100000,location='STD', ls=Run2023C)}
steps['RunDisplacedJet2023C']={'INPUT':InputInfo(dataSet='/DisplacedJet/Run2023C-v1/RAW',label='2023C',events=100000,location='STD', ls=Run2023C)}
steps['RunEGamma2023C']={'INPUT':InputInfo(dataSet='/EGamma1/Run2023C-v1/RAW',label='2023C',events=100000,location='STD', ls=Run2023C)}
steps['RunTau2023C']={'INPUT':InputInfo(dataSet='/Tau/Run2023C-v1/RAW',label='2023C',events=100000,location='STD', ls=Run2023C)}
steps['RunMuonEG2023C']={'INPUT':InputInfo(dataSet='/MuonEG/Run2023C-v1/RAW',label='2023C',events=100000,location='STD', ls=Run2023C)}
steps['RunParkingDoubleMuonLowMass2023C']={'INPUT':InputInfo(dataSet='/ParkingDoubleMuonLowMass0/Run2023C-v1/RAW',label='2023C',events=100000,location='STD', ls=Run2023C)}

Run2023C_LS500={368489: [[46,546]]}
steps['TestEnablesEcalHcal2023C']={'INPUT':InputInfo(dataSet='/TestEnablesEcalHcal/Run2023C-Express-v4/RAW',label='2023C',events=100000,location='STD', ls=Run2023C_LS500)}

Run2023D={369978: [[1, 800]]}
steps['RunMuon2023D']={'INPUT':InputInfo(dataSet='/Muon0/Run2023D-v1/RAW',label='2023D',events=100000,location='STD', ls=Run2023D)}
steps['RunZeroBias2023D']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2023D-v1/RAW',label='2023D',events=100000,location='STD', ls=Run2023D)}
steps['RunBTagMu2023D']={'INPUT':InputInfo(dataSet='/BTagMu/Run2023D-v1/RAW',label='2023D',events=100000,location='STD', ls=Run2023D)}
steps['RunJetMET2023D']={'INPUT':InputInfo(dataSet='/JetMET0/Run2023D-v1/RAW',label='2023D',events=100000,location='STD', ls=Run2023D)}
steps['RunDisplacedJet2023D']={'INPUT':InputInfo(dataSet='/DisplacedJet/Run2023D-v1/RAW',label='2023D',events=100000,location='STD', ls=Run2023D)}
steps['RunEGamma2023D']={'INPUT':InputInfo(dataSet='/EGamma0/Run2023D-v1/RAW',label='2023D',events=100000,location='STD', ls=Run2023D)}
steps['RunTau2023D']={'INPUT':InputInfo(dataSet='/Tau/Run2023D-v1/RAW',label='2023D',events=100000,location='STD', ls=Run2023D)}
steps['RunMuonEG2023D']={'INPUT':InputInfo(dataSet='/MuonEG/Run2023D-v1/RAW',label='2023D',events=100000,location='STD', ls=Run2023D)}
steps['RunParkingDoubleMuonLowMass2023D']={'INPUT':InputInfo(dataSet='/ParkingDoubleMuonLowMass0/Run2023D-v1/RAW',label='2023D',events=100000,location='STD', ls=Run2023D)}

# Highstat HLTPhysics
Run2015DHS=selectedLS([258712,258713,258714,258741,258742,258745,258749,258750,259626,259637,259683,259685,259686,259721,259809,259810,259818,259820,259821,259822,259862,259890,259891])
steps['RunHLTPhy2015DHS']={'INPUT':InputInfo(dataSet='/HLTPhysics/Run2015D-v1/RAW',label='2015DHS',events=100000,location='STD', ls=Run2015DHS)}

#### run2 2015 HighLumi High Stat workflows ##
# Run2015HLHS, 25ns, run 260627, JetHT: 2.9M, SingleMuon: 5.7M, ZeroBias: 1.6M
Run2015HLHS=selectedLS([260627])
steps['RunJetHT2015HLHS']={'INPUT':InputInfo(dataSet='/JetHT/Run2015D-v1/RAW',label='2015DHLHS',events=100000,location='STD', ls=Run2015HLHS)}
steps['RunZeroBias2015HLHS']={'INPUT':InputInfo(dataSet='/ZeroBias/Run2015D-v1/RAW',label='2015DHLHS',events=100000,location='STD', ls=Run2015HLHS)}
steps['RunSingleMu2015HLHS']={'INPUT':InputInfo(dataSet='/SingleMuon/Run2015D-v1/RAW',label='2015DHLHS',events=100000,location='STD', ls=Run2015HLHS)}

#### run2 Cosmic ####
##Run 256259 @ 0T 2015C###
##Run 272133 @ 3.8T 2016B###
steps['RunCosmics2015C']={'INPUT':InputInfo(dataSet='/Cosmics/Run2015C-v1/RAW',label='2015C',run=[256259],events=100000,location='STD')}
steps['RunCosmics2016B']={'INPUT':InputInfo(dataSet='/Cosmics/Run2016B-v1/RAW',label='2016B',run=[272133],events=100000,location='STD')}

### LS2 - MWGR 2021 - CSC, DAQ, DCS, DQM, DT, ECAL, GEM, HCAL, L1SCOUT, PIXEL, RPC, TCDS, TRACKER, TRG ###
steps['RunCosmics2021']={'INPUT':InputInfo(dataSet='/ExpressCosmics/Commissioning2021-Express-v1/FEVT',label='2021GR5',run=[343498],events=100000,location='STD')}

#### run3 Commissioning2021 ####
##Run 344518 @ 0.0T 2021CRUZET
##Run 345755 @ 3.8T 2021CRAFT
##Run 345881 @ 3.8T 2021Splash
steps['RunCosmics2021CRUZET']={'INPUT':InputInfo(dataSet='/Cosmics/Commissioning2021-v1/RAW',label='2021Commissioning',run=[344518],events=100000,location='STD')}
steps['RunCosmics2021CRAFT']={'INPUT':InputInfo(dataSet='/Cosmics/Commissioning2021-v1/RAW',label='2021Commissioning',ls={345755: [[1, 99]]},events=100000,location='STD')}
steps['RunMinimumBias2021Splash']={'INPUT':InputInfo(dataSet='/MinimumBias/Commissioning2021-v1/RAW',label='2021Commissioning',ls={345881: [782, 790, 796, 801, 1031, 1037]},events=100000,location='STD')}
##Collisions
Run2021Commissioning={346512: [[250, 300]]}
steps['RunMinimumBias2021']={'INPUT':InputInfo(dataSet='/MinimumBias/Commissioning2021-v1/RAW',label='2021Commissioning',events=100000,location='STD', ls=Run2021Commissioning)}
steps['RunZeroBias2021']={'INPUT':InputInfo(dataSet='/ZeroBias/Commissioning2021-v1/RAW',label='2021Commissioning',events=100000,location='STD', ls=Run2021Commissioning)}
steps['RunHLTPhy2021']={'INPUT':InputInfo(dataSet='/HLTPhysics/Commissioning2021-v1/RAW',label='2021Commissioning',events=100000,location='STD', ls=Run2021Commissioning)}
steps['RunNoBPTX2021']={'INPUT':InputInfo(dataSet='/NoBPTX/Commissioning2021-v1/RAW',label='2021Commissioning',events=100000,location='STD', ls=Run2021Commissioning)}
steps['AlCaPhiSym2021']={'INPUT':InputInfo(dataSet='/AlCaPhiSym/Commissioning2021-v1/RAW',label='2021Commissioning',run=[346512],location='STD')}
steps['AlCaLumiPixels2021']={'INPUT':InputInfo(dataSet='/AlCaLumiPixelsCountsExpress/Commissioning2021-v1/RAW',label='2021Commissioning',run=[346512],events=100000,location='STD')}

#### Test of lumi section boundary crossing with run2 2018D ####
Run2018Dml1={320822: [[1,1]] , 320823: [[1,1]]}
Run2018Dml2={320822: [[1,2]]}
steps['RunEGamma2018Dml1']={'INPUT':InputInfo(dataSet='/EGamma/Run2018D-v1/RAW',label='2018D',events=100000,location='STD', ls=Run2018Dml1)}
steps['RunEGamma2018Dml2']={'INPUT':InputInfo(dataSet='/EGamma/Run2018D-v1/RAW',label='2018D',events=100000,location='STD', ls=Run2018Dml2)}

#### Run2 TauEmbed ####
steps['RunDoubleMuonTE2016C']={'INPUT':InputInfo(dataSet='/DoubleMuon/Run2016C-v2/RAW',label='2016C',events=10000,location='STD', ls=Run2016C)}
dataTE = {'--conditions': 'auto:run2_data', '--scenario': 'pp'}
mcTE = {'--conditions': 'auto:run2_mc', '--scenario': 'pp'}
steps['RAWRECOTE16'] = merge([{'-s': 'RAW2DIGI,L1Reco,RECO,PAT', '--datatier': 'FEVT', '--eventcontent': 'FEVT', '--data': '', '--era': 'Run2_2016_HIPM', '--customise':'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2016,TauAnalysis/MCEmbeddingTools/customisers.customiseSelecting',}, dataTE])
steps['RAWRECOLHECLEANTE16'] = merge([{'-s': 'RAW2DIGI,RECO', '--data': '', '--eventcontent': 'RECO', '--datatier': 'RECO', '--era': 'Run2_2016_HIPM', '--customise': 'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2016,TauAnalysis/MCEmbeddingTools/customisers.customiseLHEandCleaning',}, dataTE])
steps['EMBEDHADTE16'] = merge([{'cfg': 'TauAnalysis/MCEmbeddingTools/python/EmbeddingPythia8Hadronizer_cfi.py', '-s': 'GEN,SIM,DIGI,L1,DIGI2RAW,RAW2DIGI,RECO', '--processName': 'EmbeddingPythia8Hadronizer', '--eventcontent': 'RECO', '--datatier': 'RECO', '--beamspot': 'Realistic25ns13TeV2016Collision', '--customise': 'TauAnalysis/MCEmbeddingTools/customisers.customiseGenerator', '--customise_commands': '\"process.generator.nAttempts = cms.uint32(1000)\"', '--era': 'Run2_2016_HIPM',}, mcTE])
steps['EMBEDMINIAOD16'] = merge([{'-s': 'PAT', '--data': '', '--eventcontent': 'MINIAODSIM', '--datatier': 'MINIAODSIM', '--customise': 'TauAnalysis/MCEmbeddingTools/customisers.customiseMerging', '--customise_commands': '\"process.patTrigger.processName = cms.string(\'SIMembedding\')\"', '--era': 'Run2_2016_HIPM',}, dataTE])

def gen(fragment,howMuch):
    global step1Defaults
    return merge([{'cfg':fragment},howMuch,step1Defaults])

def gen2015(fragment,howMuch):
    global step1Up2015Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2015Defaults])
def gen2017(fragment,howMuch):
    global step1Up2017Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2017Defaults])
def gen2018(fragment,howMuch):
    global step1Up2018Defaults
    return merge([{'cfg':fragment},howMuch,step1Up2018Defaults])
def gen2018prod(fragment,howMuch):
    global step1Up2018ProdDefaults
    return merge([{'cfg':fragment},howMuch,step1Up2018ProdDefaults])
def gen2018prodml(fragment,howMuch): #Prod with multiple lumis
    global step1Up2018ProdDefaults
    return merge([{'cfg':fragment},howMuch,{'--customise_commands': '"process.source.numberEventsInLuminosityBlock=cms.untracked.uint32(5)"'},step1Up2018ProdDefaults]) #this setting is to allow job to cross LS in IB
def gen2018hiprod(fragment,howMuch):
    global step1Up2018HiProdDefaults
    return merge([{'cfg':fragment},howMuch,step1Up2018HiProdDefaults])
def gen2022hiprod(fragment,howMuch):
    global step1Up2022HiProdDefaults
    return merge([{'cfg':fragment},howMuch,step1Up2022HiProdDefaults])
def gen2023hiprod(fragment,howMuch):
    global step1Up2023HiProdDefaults
    return merge([{'cfg':fragment},howMuch,step1Up2023HiProdDefaults])

### Production test: 13 TeV equivalents
steps['ProdMinBias_13']=gen2015('MinBias_13TeV_pythia8_TuneCUETP8M1_cfi',Kby(9,100))
steps['ProdTTbar_13']=gen2015('TTbar_13TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['ProdZEE_13']=gen2015('ZEE_13TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['ProdQCD_Pt_3000_3500_13']=gen2015('QCD_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi',Kby(9,100))


##production 2017
steps['ProdTTbar_13UP17']=gen2017('TTbar_13TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['ProdMinBias_13UP17']=gen2017('MinBias_13TeV_pythia8_TuneCUETP8M1_cfi',Kby(9,100))
steps['ProdQCD_Pt_3000_3500_13UP17']=gen2017('QCD_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi',Kby(9,100))



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
steps['NuGun_UP15']=gen2015('SingleNuE10_cfi.py',Kby(9,50))
steps['TTbar']=gen('TTbar_8TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['TTbarLepton']=gen('TTbarLepton_8TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['ZEE']=gen('ZEE_8TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['Wjet_Pt_80_120']=gen('Wjet_Pt_80_120_8TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['Wjet_Pt_3000_3500']=gen('Wjet_Pt_3000_3500_8TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['LM1_sfts']=gen('LM1_sfts_8TeV_cfi',Kby(9,100))
steps['QCD_FlatPt_15_3000']=gen('QCDForPF_8TeV_TuneCUETP8M1_cfi',Kby(5,100))
steps['QCD_FlatPt_15_3000HS']=gen('QCDForPF_8TeV_TuneCUETP8M1_cfi',Kby(50,100))
steps['TTbar_13']=gen2015('TTbar_13TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['TTbarLepton_13']=gen2015('TTbarLepton_13TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['ZEE_13']=gen2015('ZEE_13TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['ZEE_13_DBLMINIAOD']=gen2015('ZEE_13TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['Wjet_Pt_80_120_13']=gen2015('Wjet_Pt_80_120_13TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['Wjet_Pt_3000_3500_13']=gen2015('Wjet_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['SMS-T1tttt_mGl-1500_mLSP-100_13']=gen2015('SMS-T1tttt_mGl-1500_mLSP-100_13TeV-pythia8_cfi',Kby(9,50))
steps['QCD_FlatPt_15_3000_13']=gen2015('QCDForPF_13TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['QCD_FlatPt_15_3000HS_13']=gen2015('QCDForPF_13TeV_TuneCUETP8M1_cfi',Kby(50,100))

steps['ZpMM_2250_8TeV']=gen('ZpMM_2250_8TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['ZpEE_2250_8TeV']=gen('ZpEE_2250_8TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['ZpTT_1500_8TeV']=gen('ZpTT_1500_8TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['ZpMM_2250_13']=gen2015('ZpMM_2250_13TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['ZpEE_2250_13']=gen2015('ZpEE_2250_13TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['ZpTT_1500_13']=gen2015('ZpTT_1500_13TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['HSCPstop_M_200_13']=gen2015('HSCPstop_M_200_TuneCUETP8M1_13TeV_pythia8_cff',Kby(9,100))
steps['RSGravitonToGaGa_13']=gen2015('RSGravitonToGammaGamma_kMpl01_M_3000_TuneCUETP8M1_13TeV_pythia8_cfi',Kby(9,100))
steps['WpToENu_M-2000_13']=gen2015('WprimeToENu_M-2000_TuneCUETP8M1_13TeV-pythia8_cff',Kby(9,100))
steps['DisplacedSUSY_stopToBottom_M_300_1000mm_13']=gen2015('DisplacedSUSY_stopToBottom_M_300_1000mm_TuneCUETP8M1_13TeV_pythia8_cff',Kby(9,100))

### 2017 wf: only the ones for premixing (for the moment)
steps['NuGun_UP17']=gen2017('SingleNuE10_cfi.py',Kby(9,50))
steps['TTbar_13UP17']=gen2017('TTbar_13TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['ProdZEE_13UP17']=gen2017('ZEE_13TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['ZEE_13UP17']=gen2017('ZEE_13TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['ZMM_13UP17']=gen2017('ZMM_13TeV_TuneCUETP8M1_cfi',Kby(18,100))
steps['ZTT_13UP17']=gen2017('ZTT_All_hadronic_13TeV_TuneCUETP8M1_cfi',Kby(9,80))
steps['H125GGgluonfusion_13UP17']=gen2017('H125GGgluonfusion_13TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['QQH1352T_13UP17']=gen2017('QQH1352T_13TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['SMS-T1tttt_mGl-1500_mLSP-100_13UP17']=gen2017('SMS-T1tttt_mGl-1500_mLSP-100_13TeV-pythia8_cfi',Kby(9,50))


### 2018 wf: only the ones for premixing (for the moment)
steps['NuGun_UP18']=gen2018('SingleNuE10_cfi.py',Kby(9,50))
steps['ProdTTbar_13UP18']=gen2018prod('TTbar_13TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['ProdTTbar_13UP18ml']=gen2018prodml('TTbar_13TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['TTbar_13UP18']=gen2018('TTbar_13TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['ProdZEE_13UP18']=gen2018prod('ZEE_13TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['ProdZEE_13UP18ml']=gen2018prodml('ZEE_13TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['ZEE_13UP18']=gen2018('ZEE_13TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['ProdZMM_13UP18']=gen2018prod('ZMM_13TeV_TuneCUETP8M1_cfi',Kby(18,100))
steps['ZMM_13UP18']=gen2018('ZMM_13TeV_TuneCUETP8M1_cfi',Kby(18,100))
steps['ZTT_13UP18']=gen2018('ZTT_All_hadronic_13TeV_TuneCUETP8M1_cfi',Kby(9,80))
steps['H125GGgluonfusion_13UP18']=gen2018('H125GGgluonfusion_13TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['QQH1352T_13UP18']=gen2018('QQH1352T_13TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['SMS-T1tttt_mGl-1500_mLSP-100_13UP18']=gen2018('SMS-T1tttt_mGl-1500_mLSP-100_13TeV-pythia8_cfi',Kby(9,50))


# 13TeV High Stats samples
steps['ZMM_13_HS']=gen2015('ZMM_13TeV_TuneCUETP8M1_cfi',Kby(209,100))
steps['TTbar_13_HS']=gen2015('TTbar_13TeV_TuneCUETP8M1_cfi',Kby(100,50))


def identitySim(wf):
    return merge([{'--restoreRND':'SIM','--process':'SIM2', '--inputCommands':'"keep *","drop *TagInfo*_*_*_*"' },wf])

steps['SingleMuPt10_UP15_ID']=identitySim(steps['SingleMuPt10_UP15'])
steps['TTbar_13_ID']=identitySim(steps['TTbar_13'])

baseDataSetRelease=[
    'CMSSW_9_2_4-91X_mcRun1_realistic_v2-v1',               # 0 run1 samples; note TTbar GENSIM has v2 (see TTbarINPUT below)
    'CMSSW_10_3_0_pre5-103X_upgrade2018_realistic_v7-v1',          # 1 GEN-SIM for HI RunII, 2018
    'CMSSW_6_2_0_pre8-PRE_ST62_V8_FastSim-v1',              # 2 for fastsim id test
#    'CMSSW_7_1_0_pre5-START71_V1-v2',                      # 3 8 TeV , for the one sample which is part of the routine relval production (RelValZmumuJets_Pt_20_300, because of -v2)
                                                            # THIS ABOVE IS NOT USED, AT THE MOMENT
    'CMSSW_10_6_0-106X_mcRun2_asymptotic_v3-v1',              # 3 - GEN-SIM input for 13 TeV 2016 workfows
    'CMSSW_7_3_0_pre1-PRE_LS172_V15_FastSim-v1',                   # 4 - fast sim GEN-SIM-DIGI-RAW-HLTDEBUG for id tests
    'CMSSW_10_6_0-PU25ns_106X_mcRun2_asymptotic_v3-v1',    # 5 - fullSim PU 25ns UP15 premix library
    'CMSSW_10_4_0_pre1-PU50ns_103X_mcRun2_startup_v1-v1',        # 6 - fullSim PU 50ns UP15 premix library
    'CMSSW_10_6_0-106X_mcRun2_asymptotic_v3_FastSim-v1',    # 7 - fastSim MinBias for mixing UP16
    'CMSSW_10_6_0-PU25ns_106X_mcRun2_asymptotic_v3_FastSim-v1',# 8 - fastSim premix library  UP16
    'CMSSW_10_6_0-106X_upgrade2018_realistic_v4-v1',        # 9 - Run2 HI GEN-SIM for mixing
    'CMSSW_7_6_0-76X_mcRun2_asymptotic_v11-v1',                    # 10 - 13 TeV High Stats GEN-SIM
    'CMSSW_7_6_0_pre7-76X_mcRun2_asymptotic_v9_realBS-v1',         # 11 - 13 TeV High Stats MiniBias for mixing GEN-SIM
    'CMSSW_8_1_0_pre9_Geant4102-81X_mcRun2cosmics_startup_peak_v2-v1', # 12 - GEN-SIM input for 1307 cosmics wf from 810_p2
    'CMSSW_10_6_0-106X_mc2017_realistic_v3-v1',     # 13 - GENSIM input for 2017 fullSim premix workflows
    'CMSSW_10_6_0-PU25ns_106X_mc2017_realistic_v3-v1',     # 14 - fullSim PU 25ns UP17 premix library
    'CMSSW_10_6_0-106X_mc2017_realistic_v3_FastSim-v1',    # 15 - fastSim MinBias for mixing UP17
    'CMSSW_10_6_0-PU25ns_106X_mc2017_realistic_v3_FastSim-v1',# 16 - fastSim premix library UP17
    'CMSSW_10_6_0-PU25ns_106X_upgrade2018_realistic_v4-v1',  # 17 - fullSim PU 25ns UP18 premix library
    'CMSSW_10_6_0-106X_upgrade2018_realistic_v4-v1',  # 18 - GENSIM input for 2018 fullSim premix workflows
    'CMSSW_10_6_0-106X_upgrade2018_realistic_v4_FastSim-v1',    # 19 - fastSim MinBias for mixing UP18
    'CMSSW_10_6_0-PU25ns_106X_upgrade2018_realistic_v4_FastSim-v1',# 20 - fastSim premix library UP18
    'CMSSW_10_6_0-106X_mc2017_realistic_v3-v1',         # 21 - GEN-SIM inputs for LHE-GEN-SIM 2017 workflows
    'CMSSW_10_6_0-106X_upgrade2018_realistic_v4-v1',            # 22 - GEN-SIM inputs for LHE-GEN-SIM 2018 workflows
    'CMSSW_12_5_1-125X_mcRun3_2022_realistic_HI_v5-v1',    #23 - 2022 HI GEN-SIM for mixing
    'CMSSW_11_2_0_pre8-PU_112X_upgrade2018_realistic_v4-v1', # 24 - 2018 Run-Dependent premix library
    'CMSSW_13_2_0_pre1-131X_mcRun3_2023_realistic_HI_v11-v1', # 2023 HI GEN-SIM for mixing
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
steps['TTbarINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/%s/GEN-SIM'%((baseDataSetRelease[0].rstrip('1')+'2'),),location='STD')}
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
# INPUT command for reminiAOD wf on 80X relval input
steps['ProdZEE_13_reminiaodINPUT']={'INPUT':InputInfo(dataSet='/RelValProdZEE_13_pmx25ns/CMSSW_8_0_21-PUpmx25ns_80X_mcRun2_asymptotic_2016_TrancheIV_v6_Tr4GT_v6-v1/AODSIM',label='rmaod',location='STD')}
# INPUT command for reminiAOD wf on 94X relval input
steps['TTbar_13_94XreminiaodINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar_13/CMSSW_9_4_0-94X_mc2017_realistic_v10-v1/GEN-SIM-RECO',label='rmaod',location='STD')}
# INPUT command for reminiAOD wfs on UL-like relval inputs
steps['TTbar_13_reminiaod2016UL_preVFP_INPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar_13UP16/CMSSW_10_6_12-PU25ns_106X_mcRun2_asymptotic_preVFP_v8_hltul16_preVFP-v1/AODSIM',label='rmaod',location='STD')}
steps['TTbar_13_reminiaod2016UL_postVFP_INPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar_13UP16/CMSSW_10_6_12-PU25ns_106X_mcRun2_asymptotic_v13_hltul16_postVFP-v1/AODSIM',label='rmaod',location='STD')}
# FIXME: replace with AODSIM (more appropriate)
steps['TTbar_13_reminiaod2017UL_INPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar_13/CMSSW_10_6_4-PUpmx25ns_106X_mc2017_realistic_v6_rsb-v1/GEN-SIM-RECO',label='rmaod',location='STD')}
steps['TTbar_13_reminiaod2018UL_INPUT']={'INPUT':InputInfo(dataSet='/RelValProdTTbar_13_pmx25ns/CMSSW_10_6_4-PUpmx25ns_106X_upgrade2018_realistic_v9-v1/AODSIM',label='rmaod',location='STD')}
# INPUT command for reminiAOD wfs on PbPb relval inputs
steps['HydjetQ_reminiaodPbPb2018_INPUT']={'INPUT':InputInfo(dataSet='/RelValHydjetQ_B12_5020GeV_2018_ppReco/CMSSW_10_3_3-PU_103X_upgrade2018_realistic_HI_v11-v1/GEN-SIM-RECO',label='rmaod',location='STD')}
steps['HydjetQ_reminiaodPbPb2022_INPUT']={'INPUT':InputInfo(dataSet='/RelValHydjetQ_B12_5020GeV_2021_ppReco/CMSSW_12_5_0_pre3-124X_mcRun3_2022_realistic_HI_v8-v1/GEN-SIM-RECO',label='rmaod',location='STD')}
#steps['HydjetQ_reminiaodPbPb2023_INPUT']={'INPUT':InputInfo(dataSet='/RelValHydjetQ_B12_5362GeV_2023_ppReco/CMSSW_13_1_0_preX-131X_mcRun3_2023_realistic_HI_vX-v1/GEN-SIM-RECO',label='rmaod',location='STD')}


# 13 TeV recycle GEN-SIM input
steps['MinBias_13INPUT']={'INPUT':InputInfo(dataSet='/RelValMinBias_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['QCD_Pt_3000_3500_13INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_3000_3500_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['QCD_Pt_600_800_13INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_600_800_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['QCD_Pt_80_120_13INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_80_120_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['QCD_Pt_80_120_13_HIINPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_80_120_13_HI/%s/GEN-SIM'%(baseDataSetRelease[1],),location='STD')}
steps['TTbar_13INPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['TTbarLepton_13INPUT']={'INPUT':InputInfo(dataSet='/RelValTTbarLepton_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['ZEE_13INPUT']={'INPUT':InputInfo(dataSet='/RelValZEE_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
#steps['ZEE_13_DBLMINIAODINPUT']={'INPUT':InputInfo(dataSet='/RelValZEE_13_DBLMINIAOD/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['Wjet_Pt_80_120_13INPUT']={'INPUT':InputInfo(dataSet='/RelValWjet_Pt_80_120_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['Wjet_Pt_3000_3500_13INPUT']={'INPUT':InputInfo(dataSet='/RelValWjet_Pt_3000_3500_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['SMS-T1tttt_mGl-1500_mLSP-100_13INPUT']={'INPUT':InputInfo(dataSet='/RelValSMS-T1tttt_mGl-1500_mLSP-100_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['QCD_FlatPt_15_3000_13INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_FlatPt_15_3000_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['QCD_FlatPt_15_3000HS_13INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_FlatPt_15_3000HS_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['ZpMM_2250_13INPUT']={'INPUT':InputInfo(dataSet='/RelValZpMM_2250_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['ZpEE_2250_13INPUT']={'INPUT':InputInfo(dataSet='/RelValZpEE_2250_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['ZpTT_1500_13INPUT']={'INPUT':InputInfo(dataSet='/RelValZpTT_1500_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['MinBiasHS_13INPUT']={'INPUT':InputInfo(dataSet='/RelValMinBiasHS_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['Higgs200ChargedTaus_13INPUT']={'INPUT':InputInfo(dataSet='/RelValHiggs200ChargedTaus_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}

steps['Upsilon1SToMuMu_13INPUT']={'INPUT':InputInfo(dataSet='/RelValUpsilon1SToMuMu_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['JpsiMuMu_Pt-8INPUT']={'INPUT':InputInfo(dataSet='/RelValJpsiMuMu_Pt-8/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
# new BPH relvals produced for the first time in 810_pre9
steps['BsToMuMu_13INPUT']={'INPUT':InputInfo(dataSet='/RelValBsToMuMu_13/CMSSW_8_1_0_pre9-81X_mcRun2_asymptotic_v2-v1/GEN-SIM',location='STD')}
steps['BdToMuMu_13INPUT']={'INPUT':InputInfo(dataSet='/RelValBdToMuMu_13/CMSSW_8_1_0_pre9-81X_mcRun2_asymptotic_v2-v1/GEN-SIM',location='STD')}
steps['BuToJpsiK_13INPUT']={'INPUT':InputInfo(dataSet='/RelValBuToJpsiK_13/CMSSW_8_1_0_pre9-81X_mcRun2_asymptotic_v2-v1/GEN-SIM',location='STD')}
steps['BsToJpsiPhi_13INPUT']={'INPUT':InputInfo(dataSet='/RelValBsToJpsiPhi_13/CMSSW_8_1_0_pre9-81X_mcRun2_asymptotic_v2-v1/GEN-SIM',location='STD')}
###
steps['PhiToMuMu_13INPUT']={'INPUT':InputInfo(dataSet='/RelValPhiToMuMu_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['EtaBToJpsiJpsi_13INPUT']={'INPUT':InputInfo(dataSet='/RelValEtaBToJpsiJpsi_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['BuMixing_13INPUT']={'INPUT':InputInfo(dataSet='/RelValBuMixing_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}


steps['WE_13INPUT']={'INPUT':InputInfo(dataSet='/RelValWE_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['WM_13INPUT']={'INPUT':InputInfo(dataSet='/RelValWM_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['WpM_13INPUT']={'INPUT':InputInfo(dataSet='/RelValWpM_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['ZMM_13INPUT']={'INPUT':InputInfo(dataSet='/RelValZMM_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['ZEEMM_13_HIINPUT']={'INPUT':InputInfo(dataSet='/RelValZEEMM_13_HI/%s/GEN-SIM'%(baseDataSetRelease[1],),location='STD')}
steps['ZpMM_13INPUT']={'INPUT':InputInfo(dataSet='/RelValZpMM_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['ZTT_13INPUT']={'INPUT':InputInfo(dataSet='/RelValZTT_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['H125GGgluonfusion_13INPUT']={'INPUT':InputInfo(dataSet='/RelValH125GGgluonfusion_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['PhotonJets_Pt_10_13INPUT']={'INPUT':InputInfo(dataSet='/RelValPhotonJets_Pt_10_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['PhotonJets_Pt_10_13_HIINPUT']={'INPUT':InputInfo(dataSet='/RelValPhotonJets_Pt_10_13_HI/%s/GEN-SIM'%(baseDataSetRelease[1],),location='STD')}
steps['QQH1352T_13INPUT']={'INPUT':InputInfo(dataSet='/RelValQQH1352T_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['ADDMonoJet_d3MD3_13INPUT']={'INPUT':InputInfo(dataSet='/RelValADDMonoJet_d3MD3_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['RSKKGluon_m3000GeV_13INPUT']={'INPUT':InputInfo(dataSet='/RelValRSKKGluon_m3000GeV_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['PhiToMuMu_13']=gen2015('PYTHIA8_PhiToMuMu_TuneCUETP8M1_13TeV_cff',Kby(100,1100))
steps['EtaBToJpsiJpsi_13']=gen2015('EtaBToJpsiJpsi_forSTEAM_TuneCUEP8M1_13TeV_cfi',Kby(9,100))
steps['BuMixing_13']=gen2015('BuMixing_BMuonFilter_forSTEAM_13TeV_TuneCUETP8M1_cfi',Kby(900,10000))

steps['Cosmics_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValCosmics_UP15/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['CosmicsSPLoose_UP15INPUT']={'INPUT':InputInfo(dataSet='/RelValCosmicsSPLoose_UP15/%s/GEN-SIM'%(baseDataSetRelease[12],),location='STD')}
steps['BeamHalo_13INPUT']={'INPUT':InputInfo(dataSet='/RelValBeamHalo_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['HSCPstop_M_200_13INPUT']={'INPUT':InputInfo(dataSet='/RelValHSCPstop_M_200_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['RSGravitonToGaGa_13INPUT']={'INPUT':InputInfo(dataSet='/RelValRSGravitonToGaGa_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['WpToENu_M-2000_13INPUT']={'INPUT':InputInfo(dataSet='/RelValWpToENu_M-2000_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
steps['DisplacedSUSY_stopToBottom_M_300_1000mm_13INPUT']={'INPUT':InputInfo(dataSet='/RelValDisplacedSUSY_stopToBottom_M_300_1000mm_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}

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


# INPUT commands for 2017 wf
steps['TTbar_13UP17INPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar_13/%s/GEN-SIM'%(baseDataSetRelease[13],),location='STD')}
steps['ZEE_13UP17INPUT']={'INPUT':InputInfo(dataSet='/RelValZEE_13/%s/GEN-SIM'%(baseDataSetRelease[13],),location='STD')}
steps['ZMM_13UP17INPUT']={'INPUT':InputInfo(dataSet='/RelValZMM_13/%s/GEN-SIM'%(baseDataSetRelease[13],),location='STD')}
steps['ZTT_13UP17INPUT']={'INPUT':InputInfo(dataSet='/RelValZTT_13/%s/GEN-SIM'%(baseDataSetRelease[13],),location='STD')}
steps['H125GGgluonfusion_13UP17INPUT']={'INPUT':InputInfo(dataSet='/RelValH125GGgluonfusion_13/%s/GEN-SIM'%(baseDataSetRelease[13],),location='STD')}
steps['QQH1352T_13UP17INPUT']={'INPUT':InputInfo(dataSet='/RelValQQH1352T_13/%s/GEN-SIM'%(baseDataSetRelease[13],),location='STD')}
steps['NuGun_UP17INPUT']={'INPUT':InputInfo(dataSet='/RelValNuGun/%s/GEN-SIM'%(baseDataSetRelease[13],),location='STD')}
steps['SMS-T1tttt_mGl-1500_mLSP-100_13UP17INPUT']={'INPUT':InputInfo(dataSet='/RelValSMS-T1tttt_mGl-1500_mLSP-100_13/%s/GEN-SIM'%(baseDataSetRelease[13],),location='STD')}


# INPUT commands for 2018 wf
steps['TTbar_13UP18INPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar_13/%s/GEN-SIM'%(baseDataSetRelease[18],),location='STD')}
steps['TTbar_13UP18HEfailINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar_13/%s/GEN-SIM'%(baseDataSetRelease[18],),location='STD')}
steps['ZEE_13UP18INPUT']={'INPUT':InputInfo(dataSet='/RelValZEE_13/%s/GEN-SIM'%(baseDataSetRelease[18],),location='STD')}
steps['ZMM_13UP18INPUT']={'INPUT':InputInfo(dataSet='/RelValZMM_13/%s/GEN-SIM'%(baseDataSetRelease[18],),location='STD')}
steps['ZTT_13UP18INPUT']={'INPUT':InputInfo(dataSet='/RelValZTT_13/%s/GEN-SIM'%(baseDataSetRelease[18],),location='STD')}
steps['H125GGgluonfusion_13UP18INPUT']={'INPUT':InputInfo(dataSet='/RelValH125GGgluonfusion_13/%s/GEN-SIM'%(baseDataSetRelease[18],),location='STD')}
steps['QQH1352T_13UP18INPUT']={'INPUT':InputInfo(dataSet='/RelValQQH1352T_13/%s/GEN-SIM'%(baseDataSetRelease[18],),location='STD')}
steps['NuGun_UP18INPUT']={'INPUT':InputInfo(dataSet='/RelValNuGun/%s/GEN-SIM'%(baseDataSetRelease[18],),location='STD')}
steps['SMS-T1tttt_mGl-1500_mLSP-100_13UP18INPUT']={'INPUT':InputInfo(dataSet='/RelValSMS-T1tttt_mGl-1500_mLSP-100_13/%s/GEN-SIM'%(baseDataSetRelease[18],),location='STD')}


#input for fast sim workflows to be added - TODO

#input for 13 TeV High Stats samples
steps['ZMM_13_HSINPUT']={'INPUT':InputInfo(dataSet='/RelValZMM_13_HS/%s/GEN-SIM'%(baseDataSetRelease[10],),location='STD')}
steps['TTbar_13_HSINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar_13_HS/%s/GEN-SIM'%(baseDataSetRelease[10],),location='STD')}


## high stat step1
ecalHcal={
    '-s':'GEN,SIM,DIGI,DIGI2RAW,RAW2DIGI,L1Reco,RECO',
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
steps['BsToMuMu_13']=gen2015('BsToMuMu_13TeV_SoftQCDnonD_TuneCUEP8M1_cfi.py',Kby(21000,150000))
steps['JpsiMuMu_Pt-8']=gen2015('JpsiMuMu_Pt-8_forSTEAM_13TeV_TuneCUETP8M1_cfi',Kby(3100,100000))
steps['BdToMuMu_13']=gen2015('BdToMuMu_13TeV_SoftQCDnonD_TuneCUEP8M1_cfi',Kby(6000,60000))
steps['BuToJpsiK_13']=gen2015('BuToJpsiK_13TeV_SoftQCDnonD_TuneCUEP8M1_cfi',Kby(16000,160000))
steps['BsToJpsiPhi_13']=gen2015('BsToJpsiPhi_13TeV_SoftQCDnonD_TuneCUEP8M1_cfi',Kby(78000,400000))

steps['WE_13']=gen2015('WE_13TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['WM_13']=gen2015('WM_13TeV_TuneCUETP8M1_cfi',Kby(9,200))
steps['WpM_13']=gen2015('WpM_13TeV_TuneCUETP8M1_cfi',Kby(9,200))
steps['ZMM_13']=gen2015('ZMM_13TeV_TuneCUETP8M1_cfi',Kby(18,100))
steps['ZEEMM_13']=gen2015('ZEEMM_13TeV_TuneCUETP8M1_cfi',Kby(18,300))
steps['ZpMM_13']=gen2015('ZpMM_13TeV_TuneCUETP8M1_cfi',Kby(9,200))

steps['ZTT']=genS('ZTT_All_hadronic_8TeV_TuneCUETP8M1_cfi',Kby(9,150))
steps['H130GGgluonfusion']=genS('H130GGgluonfusion_8TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['PhotonJets_Pt_10']=genS('PhotonJet_Pt_10_8TeV_TuneCUETP8M1_cfi',Kby(9,150))
steps['QQH1352T']=genS('QQH1352T_8TeV_TuneCUETP8M1_cfi',Kby(9,100))
steps['ZTT_13']=gen2015('ZTT_All_hadronic_13TeV_TuneCUETP8M1_cfi',Kby(9,80))
steps['H125GGgluonfusion_13']=gen2015('H125GGgluonfusion_13TeV_TuneCUETP8M1_cfi',Kby(9,50))
steps['PhotonJets_Pt_10_13']=gen2015('PhotonJet_Pt_10_13TeV_TuneCUETP8M1_cfi',Kby(9,150))
steps['QQH1352T_13']=gen2015('QQH1352T_13TeV_TuneCUETP8M1_cfi',Kby(9,50))
#steps['ZmumuJets_Pt_20_300']=gen('ZmumuJets_Pt_20_300_GEN_8TeV_TuneCUETP8M1_cfg',Kby(25,100))
steps['ADDMonoJet_d3MD3']=genS('ADDMonoJet_8TeV_d3MD3_TuneCUETP8M1_cfi',Kby(9,100))
steps['ADDMonoJet_d3MD3_13']=gen2015('ADDMonoJet_13TeV_d3MD3_TuneCUETP8M1_cfi',Kby(9,100))
steps['RSKKGluon_m3000GeV_13']=gen2015('RSKKGluon_m3000GeV_13TeV_TuneCUETP8M1_cff',Kby(9,100))

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
#steps['QQH1352TINPUT']={'INPUT':InputInfo(dataSet='/RelValQQH1352T/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')} #temporary comment out
steps['ADDMonoJet_d3MD3INPUT']={'INPUT':InputInfo(dataSet='/RelValADDMonoJet_d3MD3/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['WpMINPUT']={'INPUT':InputInfo(dataSet='/RelValWpM/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['ZpMMINPUT']={'INPUT':InputInfo(dataSet='/RelValZpMM/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['ZpMM_2250_8TeVINPUT']={'INPUT':InputInfo(dataSet='/RelValZpMM_2250_8TeV/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['ZpEE_2250_8TeVINPUT']={'INPUT':InputInfo(dataSet='/RelValZpEE_2250_8TeV/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}
steps['ZpTT_1500_8TeVINPUT']={'INPUT':InputInfo(dataSet='/RelValZpTT_1500_8TeV/%s/GEN-SIM'%(baseDataSetRelease[0],),location='STD')}

steps['Cosmics']=merge([{'cfg':'UndergroundCosmicMu_cfi.py','-n':'500','--scenario':'cosmics'},Kby(666,100000),step1Defaults])
steps['CosmicsSPLoose']=merge([{'cfg':'UndergroundCosmicSPLooseMu_cfi.py','-n':'2000','--scenario':'cosmics'},Kby(5000,100000),step1Defaults])
steps['CosmicsSPLoose_UP15']=merge([{'cfg':'UndergroundCosmicSPLooseMu_cfi.py','-n':'2000','--conditions':'auto:run2_mc_cosmics','--scenario':'cosmics'},Kby(5000,500000),step1Up2015Defaults])
steps['Cosmics_UP16']=merge([{'cfg':'UndergroundCosmicMu_cfi.py','-n':'500','--conditions':'auto:run2_mc_cosmics','--scenario':'cosmics','--era':'Run2_2016'},Kby(666,100000),step1Defaults])
steps['Cosmics_UP17']=merge([{'cfg':'UndergroundCosmicMu_cfi.py','-n':'500','--conditions':'auto:phase1_2017_cosmics','--scenario':'cosmics','--era':'Run2_2017'},Kby(666,100000),step1Defaults])
steps['Cosmics_UP18']=merge([{'cfg':'UndergroundCosmicMu_cfi.py','-n':'500','--conditions':'auto:phase1_2018_cosmics','--scenario':'cosmics','--era':'Run2_2018'},Kby(666,100000),step1Defaults])
steps['Cosmics_UP21']=merge([{'cfg':'UndergroundCosmicMu_cfi.py','-n':'500','--conditions':'auto:phase1_2022_cosmics','--scenario':'cosmics','--era':'Run3'},Kby(666,100000),step1Defaults])
steps['Cosmics_UP21_0T']=merge([{'--magField':'0T','--conditions':'auto:phase1_2022_cosmics_0T'},steps['Cosmics_UP21']])
steps['CosmicsSPLoose_UP17']=merge([{'cfg':'UndergroundCosmicSPLooseMu_cfi.py','-n':'2000','--conditions':'auto:phase1_2017_cosmics','--scenario':'cosmics','--era':'Run2_2017'},Kby(5000,500000),step1Up2015Defaults])
steps['CosmicsSPLoose_UP18']=merge([{'cfg':'UndergroundCosmicSPLooseMu_cfi.py','-n':'2000','--conditions':'auto:phase1_2018_cosmics','--scenario':'cosmics','--era':'Run2_2018'},Kby(5000,500000),step1Up2015Defaults])
steps['BeamHalo']=merge([{'cfg':'BeamHalo_cfi.py','--scenario':'cosmics'},Kby(9,100),step1Defaults])
steps['BeamHalo_13']=merge([{'cfg':'BeamHalo_13TeV_cfi.py','--scenario':'cosmics'},Kby(9,100),step1Up2015Defaults])
steps['BeamHalo_UP18']=merge([{'cfg':'BeamHalo_13TeV_cfi.py','-n':'500','--conditions':'auto:phase1_2018_cosmics','--scenario':'cosmics','--era':'Run2_2018','--beamspot':'Realistic25ns13TeVEarly2018Collision'},Kby(666,100000),step1Defaults])
steps['BeamHalo_UP21']=merge([{'cfg':'BeamHalo_13TeV_cfi.py','-n':'500','--conditions':'auto:phase1_2022_cosmics','--scenario':'cosmics','--era':'Run3','--beamspot':'Realistic25ns13TeVEarly2017Collision'},Kby(666,100000),step1Defaults])

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

## pPb Run2
step1PPbDefaultsUp15={'--beamspot':'RealisticPPbBoost8TeV2016Collision','--conditions':'auto:run2_mc_pa','--eventcontent':'RAWSIM', '--era':'Run2_2016_pA'}
steps['EPOS_PPb_8160GeV_MinimumBias']=merge([{'-n':10},step1PPbDefaultsUp15,gen2015('ReggeGribovPartonMC_EposLHC_4080_4080GeV_pPb_cfi',Kby(9,100))])

## heavy ions tests
U2000by1={'--relval': '2000,1'}
U80by1={'--relval': '80,1'}

hiDefaults2011 = {'--conditions':'auto:run1_mc_hi','--scenario':'HeavyIons'}
hiDefaults2015 = {'--conditions':'auto:run2_mc_hi', '--era':'Run2_HI','--scenario':'HeavyIons'}
hiDefaults2017 = {'--conditions':'auto:phase1_2017_realistic', '--era':'Run2_2017_pp_on_XeXe'}
hiDefaults2018 = {'--conditions':'auto:phase1_2018_realistic_hi', '--era':'Run2_2018','--scenario':'HeavyIons'}
hiDefaults2018_ppReco = {'--conditions':'auto:phase1_2018_realistic_hi', '--era':'Run2_2018_pp_on_AA'}
hiDefaults2022_ppReco = {'--conditions':'auto:phase1_2022_realistic_hi', '--era':'Run3_pp_on_PbPb'}
hiDefaults2022_ppReco_approxClusters = {'--conditions':'auto:phase1_2022_realistic_hi', '--era':'Run3_pp_on_PbPb_approxSiStripClusters'}
hiDefaults2023_ppReco = {'--conditions':'auto:phase1_2023_realistic_hi', '--era':'Run3_pp_on_PbPb_2023'}
hiDefaults2023_ppReco_approxClusters = {'--conditions':'auto:phase1_2023_realistic_hi', '--era':'Run3_pp_on_PbPb_approxSiStripClusters_2023'}

steps['Hydjet2Q_MinBias_5020GeV_2018_ppReco']=merge([{'-n':1},hiDefaults2018_ppReco,gen2018hiprod('Hydjet2_Quenched_MinBias_5020GeV_cfi',U2000by1)])
steps['HydjetQ_B12_5020GeV_2011']=merge([{'-n':1,'--beamspot':'RealisticHI2011Collision'},hiDefaults2011,genS('Hydjet_Quenched_B12_5020GeV_cfi',U2000by1)])
steps['HydjetQ_B12_5020GeV_2015']=merge([{'-n':1,'--beamspot':'RealisticHICollisionFixZ2015'},hiDefaults2015,genS('Hydjet_Quenched_B12_5020GeV_cfi',U2000by1)])
steps['HydjetQ_MinBias_XeXe_5442GeV_2017']=merge([{'-n':1},hiDefaults2017,gen2017('Hydjet_Quenched_MinBias_XeXe_5442GeV_cfi',U2000by1)])
steps['HydjetQ_B12_5020GeV_2018']=merge([{'-n':1},hiDefaults2018,gen2018prod('Hydjet_Quenched_B12_5020GeV_cfi',U2000by1)])
steps['HydjetQ_B12_5020GeV_2018_ppReco']=merge([{'-n':1},hiDefaults2018_ppReco,gen2018hiprod('Hydjet_Quenched_B12_5020GeV_cfi',U2000by1)])
steps['HydjetQ_B12_5020GeV_2022_ppReco']=merge([{'-n':1},hiDefaults2022_ppReco,gen2022hiprod('Hydjet_Quenched_B12_5020GeV_cfi',U2000by1)])
steps['HydjetQ_MinBias_5020GeV_2022_ppReco']=merge([{'-n':1},hiDefaults2022_ppReco,gen2022hiprod('Hydjet_Quenched_MinBias_5020GeV_cfi',U2000by1)])
steps['HydjetQ_B12_5362GeV_2023_ppReco']=merge([{'-n':1},hiDefaults2023_ppReco,gen2023hiprod('Hydjet_Quenched_B12_5362GeV_cfi',U2000by1)])
steps['HydjetQ_MinBias_5362GeV_2023_ppReco']=merge([{'-n':1},hiDefaults2023_ppReco,gen2023hiprod('Hydjet_Quenched_MinBias_5362GeV_cfi',U2000by1)])

steps['QCD_Pt_80_120_13_HI']=merge([hiDefaults2018_ppReco,gen2018hiprod('QCD_Pt_80_120_13TeV_TuneCUETP8M1_cfi',Kby(9,150))])
steps['PhotonJets_Pt_10_13_HI']=merge([hiDefaults2018_ppReco,gen2018hiprod('PhotonJet_Pt_10_13TeV_TuneCUETP8M1_cfi',Kby(9,150))])
steps['ZEEMM_13_HI']=merge([hiDefaults2018_ppReco,gen2018hiprod('ZEEMM_13TeV_TuneCUETP8M1_cfi',Kby(18,300))])

steps['QCD_Pt_80_120_14_HI_2022']=merge([hiDefaults2022_ppReco,gen2022hiprod('QCD_Pt_80_120_14TeV_TuneCP5_cfi',Kby(9,150))])
steps['PhotonJets_Pt_10_14_HI_2022']=merge([hiDefaults2022_ppReco,gen2022hiprod('PhotonJet_Pt_10_14TeV_TuneCP5_cfi',Kby(9,150))])
steps['ZMM_14_HI_2022']=merge([hiDefaults2022_ppReco,gen2022hiprod('ZMM_14TeV_TuneCP5_cfi',Kby(18,300))])
steps['ZEE_14_HI_2022']=merge([hiDefaults2022_ppReco,gen2022hiprod('ZEE_14TeV_TuneCP5_cfi',Kby(18,300))])

steps['QCD_Pt_80_120_5362_HI_2023']=merge([hiDefaults2023_ppReco,gen2023hiprod('QCD_Pt_80_120_5362GeV_TuneCP5_cfi',Kby(9,150))])
steps['PhotonJets_Pt_10_5362_HI_2023']=merge([hiDefaults2023_ppReco,gen2023hiprod('PhotonJet_Pt_10_5362GeV_TuneCP5_cfi',Kby(9,150))])
steps['ZMM_5362_HI_2023']=merge([hiDefaults2023_ppReco,gen2023hiprod('ZMM_5362GeV_TuneCP5_cfi',Kby(18,300))])
steps['ZEE_5362_HI_2023']=merge([hiDefaults2023_ppReco,gen2023hiprod('ZEE_5362GeV_TuneCP5_cfi',Kby(18,300))])

## pp reference tests

ppRefAlca2017 = {'--conditions':'auto:phase1_2017_realistic_ppref', '--era':'Run2_2017_ppRef', '--beamspot':'Fixed_EmitRealistic5TeVppCollision2017'}
ppRefDefaults2017=merge([ppRefAlca2017,{'-n':2}])

steps['QCD_Pt_80_120_13_PPREF']=merge([ppRefDefaults2017,gen2017('QCD_Pt_80_120_13TeV_TuneCUETP8M1_cfi',Kby(9,150))])

#### fastsim section ####
##no forseen to do things in two steps GEN-SIM then FASTIM->end: maybe later
#step1FastDefaults =merge([{'-s':'GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,L1,DIGI2RAW,L1Reco,RECO,HLT:@fake,VALIDATION:@standardValidation,DQM:@standardDQM',
step1FastDefaults =merge([{'-s':'GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,L1,DIGI2RAW,L1Reco,RECO,VALIDATION:@standardValidation,DQM:@standardDQMFS',
                           '--fast':'',
                           '--beamspot'    : 'Realistic8TeVCollision',
                           '--eventcontent':'FEVTDEBUGHLT,DQM',
                           '--datatier':'GEN-SIM-DIGI-RECO,DQMIO',
                           '--relval':'27000,3000'},
                          step1Defaults])
#step1FastUpg2015Defaults =merge([{'-s':'GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,L1,DIGI2RAW,L1Reco,RECO,HLT:@relval2016,VALIDATION:@standardValidation,DQM:@standardDQM',
step1FastUpg2015Defaults =merge([{'-s':'GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,L1,DIGI2RAW,L1Reco,RECO,VALIDATION:@standardValidation,DQM:@standardDQMFS',
                           '--fast':'',
                           '--conditions'  :'auto:run2_mc',
                           '--beamspot'    : 'Realistic25ns13TeV2016Collision',
                           '--era'         :'Run2_2016',
                           '--eventcontent':'FEVTDEBUGHLT,DQM',
                           '--datatier':'GEN-SIM-DIGI-RECO,DQMIO',
                           '--relval':'27000,3000'},
                           step1Defaults])
step1FastPUNewMixing =merge([{'-s':'GEN,SIM,RECOBEFMIX',
                           '--eventcontent':'FASTPU',
                           '--datatier':'GEN-SIM-RECO'},
                           step1FastUpg2015Defaults])
step1FastUpg2015_trackingOnlyValidation = merge([{'-s':'GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,L1,DIGI2RAW,RECO,VALIDATION:@trackingOnlyValidation'},
                                                step1FastUpg2015Defaults])

step1FastUpg2017Defaults =merge([{'-s':'GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,L1,DIGI2RAW,L1Reco,RECO,VALIDATION:@standardValidation,DQM:@standardDQMFS',
                           '--fast':'',
                           '--conditions'  :'auto:phase1_2017_realistic',
                           '--beamspot'    : 'Realistic25ns13TeVEarly2017Collision',
                           '--era'         :'Run2_2017_FastSim',
                           '--eventcontent':'FEVTDEBUGHLT,DQM',
                           '--datatier':'GEN-SIM-DIGI-RECO,DQMIO',
                           '--relval':'27000,3000'},
                           step1Defaults])
step1FastPU17NewMixing =merge([{'-s':'GEN,SIM,RECOBEFMIX',
                           '--eventcontent':'FASTPU',
                           '--datatier':'GEN-SIM-RECO'},
                           step1FastUpg2017Defaults])
step1FastUpg2017_trackingOnlyValidation = merge([{'-s':'GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,L1,DIGI2RAW,RECO,VALIDATION:@trackingOnlyValidation'},
                                                step1FastUpg2017Defaults])

step1FastUpg2018Defaults =merge([{'-s':'GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,L1,DIGI2RAW,L1Reco,RECO,VALIDATION:@standardValidation,DQM:@standardDQMFS',
                           '--fast':'',
                           '--conditions'  :'auto:phase1_2018_realistic',
                           '--beamspot'    :'Realistic25ns13TeVEarly2018Collision',
                           '--era'         :'Run2_2018_FastSim',
                           '--eventcontent':'FEVTDEBUGHLT,DQM',
                           '--datatier':'GEN-SIM-DIGI-RECO,DQMIO',
                           '--relval':'27000,3000'},
                           step1Defaults])
step1FastPU18NewMixing =merge([{'-s':'GEN,SIM,RECOBEFMIX',
                           '--eventcontent':'FASTPU',
                           '--datatier':'GEN-SIM-RECO'},
                           step1FastUpg2018Defaults])
step1FastUpg2018_trackingOnlyValidation = merge([{'-s':'GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,L1,DIGI2RAW,RECO,VALIDATION:@trackingOnlyValidation'},
                                                step1FastUpg2018Defaults])

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
steps['TTbarFS_13_trackingOnlyValidation']=merge([{'cfg':'TTbar_13TeV_TuneCUETP8M1_cfi'},Kby(100,1000),step1FastUpg2015_trackingOnlyValidation])
steps['SMS-T1tttt_mGl-1500_mLSP-100FS_13']=merge([{'cfg':'SMS-T1tttt_mGl-1500_mLSP-100_13TeV-pythia8_cfi'},Kby(100,1000),step1FastUpg2015Defaults])
steps['NuGunFS_UP15']=merge([{'cfg':'SingleNuE10_cfi'},Kby(100,1000),step1FastUpg2015Defaults])
steps['ZEEFS_13']=merge([{'cfg':'ZEE_13TeV_TuneCUETP8M1_cfi'},Kby(100,2000),step1FastUpg2015Defaults])
steps['ZTTFS_13']=merge([{'cfg':'ZTT_All_hadronic_13TeV_TuneCUETP8M1_cfi'},Kby(100,2000),step1FastUpg2015Defaults])
steps['ZMMFS_13']=merge([{'cfg':'ZMM_13TeV_TuneCUETP8M1_cfi'},Kby(100,2000),step1FastUpg2015Defaults])
steps['QCDFlatPt153000FS_13']=merge([{'cfg':'QCDForPF_13TeV_TuneCUETP8M1_cfi'},Kby(27,2000),step1FastUpg2015Defaults])
steps['QCD_Pt_80_120FS_13']=merge([{'cfg':'QCD_Pt_80_120_13TeV_TuneCUETP8M1_cfi'},Kby(100,500),step1FastUpg2015Defaults])
steps['QCD_Pt_3000_3500FS_13']=merge([{'cfg':'QCD_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi'},Kby(100,500),step1FastUpg2015Defaults])
steps['H125GGgluonfusionFS_13']=merge([{'cfg':'H125GGgluonfusion_13TeV_TuneCUETP8M1_cfi'},step1FastUpg2015Defaults])
steps['SingleMuPt10FS_UP15']=merge([{'cfg':'SingleMuPt10_pythia8_cfi'},step1FastUpg2015Defaults])
steps['SingleMuPt100FS_UP15']=merge([{'cfg':'SingleMuPt100_pythia8_cfi'},step1FastUpg2015Defaults])

### FastSim: produce sample of minbias events for PU mixing
steps['MinBiasFS_13_ForMixing']=merge([{'cfg':'MinBias_13TeV_pythia8_TuneCUETP8M1_cfi'},Kby(100,1000),step1FastPUNewMixing])

#step1FastUpg2017Defaults
steps['TTbarFS_13_UP17']=merge([{'cfg':'TTbar_13TeV_TuneCUETP8M1_cfi'},Kby(100,1000),step1FastUpg2017Defaults])
steps['TTbarFS_13_trackingOnlyValidation_UP17']=merge([{'cfg':'TTbar_13TeV_TuneCUETP8M1_cfi'},Kby(100,1000),step1FastUpg2017_trackingOnlyValidation])
steps['SMS-T1tttt_mGl-1500_mLSP-100FS_13_UP17']=merge([{'cfg':'SMS-T1tttt_mGl-1500_mLSP-100_13TeV-pythia8_cfi'},Kby(100,1000),step1FastUpg2017Defaults])
steps['ZEEFS_13_UP17']=merge([{'cfg':'ZEE_13TeV_TuneCUETP8M1_cfi'},Kby(100,2000),step1FastUpg2017Defaults])
steps['ZTTFS_13_UP17']=merge([{'cfg':'ZTT_All_hadronic_13TeV_TuneCUETP8M1_cfi'},Kby(100,2000),step1FastUpg2017Defaults])
steps['ZMMFS_13_UP17']=merge([{'cfg':'ZMM_13TeV_TuneCUETP8M1_cfi'},Kby(100,2000),step1FastUpg2017Defaults])
steps['QCDFlatPt153000FS_13_UP17']=merge([{'cfg':'QCDForPF_13TeV_TuneCUETP8M1_cfi'},Kby(27,2000),step1FastUpg2017Defaults])
steps['QCD_Pt_80_120FS_13_UP17']=merge([{'cfg':'QCD_Pt_80_120_13TeV_TuneCUETP8M1_cfi'},Kby(100,500),step1FastUpg2017Defaults])
steps['H125GGgluonfusionFS_13_UP17']=merge([{'cfg':'H125GGgluonfusion_13TeV_TuneCUETP8M1_cfi'},step1FastUpg2017Defaults])
steps['SingleMuPt10FS_UP17']=merge([{'cfg':'SingleMuPt10_pythia8_cfi'},step1FastUpg2017Defaults])
steps['SingleMuPt100FS_UP17']=merge([{'cfg':'SingleMuPt100_pythia8_cfi'},step1FastUpg2017Defaults])

### FastSim: produce sample of minbias events for PU mixing
steps['MinBiasFS_13_UP17_ForMixing']=merge([{'cfg':'MinBias_13TeV_pythia8_TuneCUETP8M1_cfi'},Kby(100,1000),step1FastPU17NewMixing])

#step1FastUpg2018Defaults
steps['TTbarFS_13_UP18']=merge([{'cfg':'TTbar_13TeV_TuneCUETP8M1_cfi'},Kby(100,1000),step1FastUpg2018Defaults])
steps['TTbarFS_13_trackingOnlyValidation_UP18']=merge([{'cfg':'TTbar_13TeV_TuneCUETP8M1_cfi'},Kby(100,1000),step1FastUpg2018_trackingOnlyValidation])
steps['SMS-T1tttt_mGl-1500_mLSP-100FS_13_UP18']=merge([{'cfg':'SMS-T1tttt_mGl-1500_mLSP-100_13TeV-pythia8_cfi'},Kby(100,1000),step1FastUpg2018Defaults])
steps['ZEEFS_13_UP18']=merge([{'cfg':'ZEE_13TeV_TuneCUETP8M1_cfi'},Kby(100,2000),step1FastUpg2018Defaults])
steps['ZTTFS_13_UP18']=merge([{'cfg':'ZTT_All_hadronic_13TeV_TuneCUETP8M1_cfi'},Kby(100,2000),step1FastUpg2018Defaults])
steps['ZMMFS_13_UP18']=merge([{'cfg':'ZMM_13TeV_TuneCUETP8M1_cfi'},Kby(100,2000),step1FastUpg2018Defaults])
steps['QCDFlatPt153000FS_13_UP18']=merge([{'cfg':'QCDForPF_13TeV_TuneCUETP8M1_cfi'},Kby(27,2000),step1FastUpg2018Defaults])
steps['QCD_Pt_80_120FS_13_UP18']=merge([{'cfg':'QCD_Pt_80_120_13TeV_TuneCUETP8M1_cfi'},Kby(100,500),step1FastUpg2018Defaults])
steps['H125GGgluonfusionFS_13_UP18']=merge([{'cfg':'H125GGgluonfusion_13TeV_TuneCUETP8M1_cfi'},step1FastUpg2018Defaults])
steps['SingleMuPt10FS_UP18']=merge([{'cfg':'SingleMuPt10_pythia8_cfi'},step1FastUpg2018Defaults])
steps['SingleMuPt100FS_UP18']=merge([{'cfg':'SingleMuPt100_pythia8_cfi'},step1FastUpg2018Defaults])

### FastSim: produce sample of minbias events for PU mixing, 2018
steps['MinBiasFS_13_UP18_ForMixing']=merge([{'cfg':'MinBias_13TeV_pythia8_TuneCUETP8M1_cfi'},Kby(100,1000),step1FastPU18NewMixing])

### FastSim: template to produce signal and overlay with minbias events
PUFS25={'--pileup':'AVE_35_BX_25ns',
        '--pileup_input':'das:/RelValMinBiasFS_13_ForMixing/%s/GEN-SIM-RECO'%(baseDataSetRelease[7],)}
FS_UP15_PU25_OVERLAY = merge([PUFS25,Kby(100,500),steps['TTbarFS_13']] )

### FastSim: template to produce signal and overlay with minbias events #PU50
PUFSAVE50={'--pileup':'AVE_50_BX_25ns',
        '--pileup_input':'das:/RelValMinBiasFS_13_UP17_ForMixing/%s/GEN-SIM-RECO'%(baseDataSetRelease[15],)}
FS_UP17_PU50_OVERLAY = merge([PUFSAVE50,Kby(100,500),steps['TTbarFS_13_UP17']] )

### FastSim: template to produce signal and overlay with minbias events #PU50, 2018
PUFSAVE50UP18={'--pileup':'AVE_50_BX_25ns',
        '--pileup_input':'das:/RelValMinBiasFS_13_UP18_ForMixing/%s/GEN-SIM-RECO'%(baseDataSetRelease[19],)}
FS_UP18_PU50_OVERLAY = merge([PUFSAVE50UP18,Kby(100,500),steps['TTbarFS_13_UP18']] )

### FastSim: produce sample of premixed minbias events
steps["FS_PREMIXUP15_PU25"] = merge([
        {"cfg":"SingleNuE10_cfi",
         "--fast":"",
         "--conditions":"auto:run2_mc",
         "-s":"GEN,SIM,RECOBEFMIX,DIGI",
         "--eventcontent":"PREMIX",
         "--datatier":"PREMIX",
         "--procModifiers":"premix_stage1",
         "--era":"Run2_2016",
         },
        PUFS25,Kby(100,500)])

### Fastsim: template to produce signal and overlay it with premixed minbias events
FS_PREMIXUP15_PU25_OVERLAY = merge([
#        {"-s" : "GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,DATAMIX,L1,DIGI2RAW,L1Reco,RECO,HLT:@relval2016,VALIDATION",
        {"-s" : "GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,DATAMIX,L1,DIGI2RAW,L1Reco,RECO,VALIDATION",
         "--datamix" : "PreMix",
         "--procModifiers": "premix_stage2",
         "--pileup_input" : "dbs:/RelValFS_PREMIXUP15_PU25/%s/PREMIX"%(baseDataSetRelease[8],),
         },
        Kby(100,500),step1FastUpg2015Defaults])

# For combined premixing stage1+stage2 workflow
FS_PREMIXUP15_PU25_LOCAL_OVERLAY = merge([
        {"--pileup_input": "file:step1.root"
         },
        FS_PREMIXUP15_PU25_OVERLAY
])

### FastSim: list of processes used in FastSim validation
fs_proclist = ["ZEE_13",'TTbar_13','H125GGgluonfusion_13','ZTT_13','ZMM_13','NuGun_UP15','QCD_FlatPt_15_3000HS_13','SMS-T1tttt_mGl-1500_mLSP-100_13']

### FastSim: produces sample of signal events, overlayed with premixed minbias events
for x in fs_proclist:
    key = "FS_" + x + "_PRMXUP15_PU25"
    steps[key] = merge([FS_PREMIXUP15_PU25_OVERLAY,{"cfg":steps[x]["cfg"]}])

    key = key.replace("PRMXUP15", "PRMXLOCALUP15")
    steps[key] = merge([FS_PREMIXUP15_PU25_LOCAL_OVERLAY,{"cfg":steps[x]["cfg"]}])

### FastSim: produce sample of signal events, overlayed with minbias events
for x in fs_proclist:
    key = "FS_" + x + "_UP15_PU25"
    steps[key] = merge([{"cfg":steps[x]["cfg"]},FS_UP15_PU25_OVERLAY])

### FastSim: produce sample of premixed minbias events UP17
steps["FS_PREMIXUP17_PU50"] = merge([
        {"cfg":"SingleNuE10_cfi",
         "--fast":"",
         "--conditions":"auto:phase1_2017_realistic",
         "-s":"GEN,SIM,RECOBEFMIX,DIGI",
         "--eventcontent":"PREMIX",
         "--datatier":"PREMIX",
         "--procModifiers":"premix_stage1",
         "--era":"Run2_2017_FastSim",
         },
        PUFSAVE50,Kby(100,500)])

### Fastsim: template to produce signal and overlay it with premixed minbias events
FS_PREMIXUP17_PU50_OVERLAY = merge([
#        {"-s" : "GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,DATAMIX,L1,DIGI2RAW,L1Reco,RECO,HLT:@relval2016,VALIDATION",
        {"-s" : "GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,DATAMIX,L1,DIGI2RAW,L1Reco,RECO,VALIDATION",
         "--datamix" : "PreMix",
         "--procModifiers": "premix_stage2",
         "--pileup_input" : "dbs:/RelValFS_PREMIXUP17_PU50/%s/PREMIX"%(baseDataSetRelease[16],),
         },
        Kby(100,500),step1FastUpg2017Defaults])

# For combined premixing stage1+stage2 workflow
FS_PREMIXUP17_PU50_LOCAL_OVERLAY = merge([
        {"--pileup_input": "file:step1.root"
         },
        FS_PREMIXUP17_PU50_OVERLAY
])

### FastSim: list of processes used in FastSim validation
fs_proclist = ["ZEE_13",'TTbar_13','H125GGgluonfusion_13','ZTT_13','ZMM_13','NuGun_UP17','QCD_FlatPt_15_3000HS_13','SMS-T1tttt_mGl-1500_mLSP-100_13']

### FastSim: produces sample of signal events, overlayed with premixed minbias events
for x in fs_proclist:
    key = "FS_" + x + "_PRMXUP17_PU50"
    steps[key] = merge([FS_PREMIXUP17_PU50_OVERLAY,{"cfg":steps[x]["cfg"]}])

    key = key.replace("PRMXUP17", "PRMXLOCALUP17")
    steps[key] = merge([FS_PREMIXUP17_PU50_LOCAL_OVERLAY,{"cfg":steps[x]["cfg"]}])

### FastSim: produce sample of signal events, overlayed with minbias events
for x in fs_proclist:
    key = "FS_" + x + "_UP17_PU50"
    steps[key] = merge([{"cfg":steps[x]["cfg"]},FS_UP17_PU50_OVERLAY])

###end UP17
###

### FastSim: produce sample of premixed minbias events UP18
steps["FS_PREMIXUP18_PU50"] = merge([
        {"cfg":"SingleNuE10_cfi",
         "--fast":"",
         "--conditions":"auto:phase1_2018_realistic",
         "-s":"GEN,SIM,RECOBEFMIX,DIGI",
         "--eventcontent":"PREMIX",
         "--datatier":"PREMIX",
         "--procModifiers":"premix_stage1",
         "--era":"Run2_2018_FastSim",
         },
        PUFSAVE50UP18,Kby(100,500)])

### Fastsim: template to produce signal and overlay it with premixed minbias events
FS_PREMIXUP18_PU50_OVERLAY = merge([
#        {"-s" : "GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,DATAMIX,L1,DIGI2RAW,L1Reco,RECO,HLT:@relval2016,VALIDATION",
        {"-s" : "GEN,SIM,RECOBEFMIX,DIGI:pdigi_valid,DATAMIX,L1,DIGI2RAW,L1Reco,RECO,VALIDATION",
         "--datamix" : "PreMix",
         "--procModifiers": "premix_stage2",
         "--pileup_input" : "dbs:/RelValFS_PREMIXUP18_PU50/%s/PREMIX"%(baseDataSetRelease[20],),
         },
        Kby(100,500),step1FastUpg2018Defaults])

# For combined premixing stage1+stage2 workflow
FS_PREMIXUP18_PU50_LOCAL_OVERLAY = merge([
        {"--pileup_input": "file:step1.root"
         },
        FS_PREMIXUP18_PU50_OVERLAY
])

### FastSim: list of processes used in FastSim validation
fs_proclist = ["ZEE_13",'TTbar_13','H125GGgluonfusion_13','ZTT_13','ZMM_13','NuGun_UP18','QCD_FlatPt_15_3000HS_13','SMS-T1tttt_mGl-1500_mLSP-100_13']

### FastSim: produces sample of signal events, overlayed with premixed minbias events
for x in fs_proclist:
    key = "FS_" + x + "_PRMXUP18_PU50"
    steps[key] = merge([FS_PREMIXUP18_PU50_OVERLAY,{"cfg":steps[x]["cfg"]}])

    key = key.replace("PRMXUP18", "PRMXLOCALUP18")
    steps[key] = merge([FS_PREMIXUP18_PU50_LOCAL_OVERLAY,{"cfg":steps[x]["cfg"]}])

### FastSim: produce sample of signal events, overlayed with minbias events
for x in fs_proclist:
    key = "FS_" + x + "_UP18_PU50"
    steps[key] = merge([{"cfg":steps[x]["cfg"]},FS_UP18_PU50_OVERLAY])

###end UP18
###


steps['TTbarSFS']=merge([{'cfg':'TTbar_8TeV_TuneCUETP8M1_cfi'},
                        {'-s':'GEN,SIM',
                         '--eventcontent':'FEVTDEBUG',
                         '--datatier':'GEN-SIM',
                         '--fast':''},
                        step1Defaults])

steps['TTbarSFSA']=merge([{'cfg':'TTbar_8TeV_TuneCUETP8M1_cfi',
#                           '-s':'GEN,SIM,RECO,HLT:@fake,VALIDATION',
                           '-s':'GEN,SIM,RECO,VALIDATION',
                           '--fast':''},
                          step1FastDefaults])

def identityFS(wf):
    return merge([{'--restoreRND':'HLT','--process':'HLT2','--hltProcess':'HLT2', '--inputCommands':'"keep *","drop *TagInfo*_*_*_*"'},wf])

steps['SingleMuPt10FS_UP15_ID']=identityFS(steps['SingleMuPt10FS_UP15'])
steps['TTbarFS_13_ID']=identityFS(steps['TTbarFS_13'])

## GENERATORS

step1GenDefaults=merge([{'-s':'GEN,VALIDATION:genvalid',
                         '--relval':'250000,5000',
                         '--eventcontent':'RAWSIM,DQM',
                         '--datatier':'GEN,DQMIO',
                         '--conditions':'auto:run2_mc'
                         },
                        step1Defaults])

step1HadronizerDefaults=merge([{'--datatier':'GEN-SIM,DQMIO',
                           '--relval':'200000,5000'
                            },step1GenDefaults])

step1LHEDefaults=merge([{'-s':'LHE',
                         '--relval':'200000,5000',
                         '--eventcontent':'LHE',
                         '--datatier':'GEN',
                         '--conditions':'auto:run2_mc'
                         },
                        step1Defaults])

steps['DYToll01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV']=genvalid('Configuration/Generator/python/DYToll01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV_cff.py',step1LHEDefaults)
steps['DYToll012Jets_5f_LO_MLM_Madgraph_LHE_13TeV']=genvalid('Configuration/Generator/python/DYToll012Jets_5f_LO_MLM_Madgraph_LHE_13TeV_cff.py',step1LHEDefaults)
steps['TTbar012Jets_5f_NLO_FXFX_Madgraph_LHE_13TeV']=genvalid('Configuration/Generator/python/TTbar012Jets_5f_NLO_FXFX_Madgraph_LHE_13TeV_cff.py',step1LHEDefaults)
steps['TTbar_Pow_LHE_13TeV']=genvalid('Configuration/Generator/python/TTbar_Pow_LHE_13TeV_cff.py',step1LHEDefaults)
steps['DYToll012Jets_5f_NLO_FXFX_Madgraph_LHE_13TeV']=genvalid('Configuration/Generator/python/DYToll012Jets_5f_NLO_FXFX_Madgraph_LHE_13TeV_cff.py',step1LHEDefaults)
steps['WTolNu01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV']=genvalid('Configuration/Generator/python/WTolNu01234Jets_5f_LO_MLM_Madgraph_LHE_13TeV_cff.py',step1LHEDefaults)
steps['WTolNu012Jets_5f_LO_MLM_Madgraph_LHE_13TeV']=genvalid('Configuration/Generator/python/WTolNu012Jets_5f_LO_MLM_Madgraph_LHE_13TeV_cff.py',step1LHEDefaults)
steps['GGToH_Pow_LHE_13TeV']=genvalid('Configuration/Generator/python/GGToH_Pow_LHE_13TeV_cff.py',step1LHEDefaults)
steps['VHToH_Pow_LHE_13TeV']=genvalid('Configuration/Generator/python/VHToH_Pow_LHE_13TeV_cff.py',step1LHEDefaults)
steps['VBFToH_Pow_JHU4l_LHE_13TeV']=genvalid('Configuration/Generator/python/VBFToH_Pow_JHU4l_LHE_13TeV_cff.py',step1LHEDefaults)

steps['BulkG_M1200_narrow_2L2Q_LHE_13TeV']=genvalid('Configuration/Generator/python/BulkG_M1200_narrow_2L2Q_LHE_13TeV_cff.py',step1LHEDefaults)

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

steps['QCD_Pt-30_13TeV_aMCatNLO_herwig7']=genvalid('Herwig7_Matchbox_aMCatNLO_Herwig_ppTojj_cff',step1GenDefaults)

steps['ZprimeToll_M3000_13TeV_pythia8']=genvalid('ZprimeToll_M3000_13TeV_pythia8_cff',step1GenDefaults)
steps['WprimeTolNu_M3000_13TeV_pythia8']=genvalid('WprimeTolNu_M3000_13TeV_pythia8_cff',step1GenDefaults)

steps['BsToMuMu_forSTEAM_13TeV_ExtGen']=genvalid('BsToMuMu_forSTEAM_13TeV_ExtGen_cfi',step1GenDefaults)
steps['BuToKstarJPsiToMuMu_forSTEAM_13TeV_ExtGen']=genvalid('BuToKstarJPsiToMuMu_forSTEAM_13TeV_ExtGen_cfi',step1GenDefaults)
steps['ZTTFS_ExtGen']=genvalid('ZTT_Tauola_OneLepton_OtherHadrons_8TeV_TuneCUETP8M1_ExtGen_cfi',step1GenDefaults)
steps['sherpa_ttbar_2j_MENLOPS_13TeV_MASTER_ExtGen']=genvalid('sherpa_ttbar_2j_MENLOPS_13TeV_MASTER_ExtGen_cff',merge([{'-n':'50'},step1GenDefaults]))
steps['HydjetQ_B12_5020GeV_2018_ExtGen']=genvalid('Hydjet_Quenched_B12_5020GeV_ExtGen_cfi',step1GenDefaults)
steps['AMPT_PPb_5020GeV_MinimumBias_ExtGen']=genvalid('AMPT_PPb_5020GeV_MinimumBias_ExtGen_cfi',step1GenDefaults)
steps['EPOS_PPb_8160GeV_MinimumBias_ExtGen']=genvalid('ReggeGribovPartonMC_EposLHC_4080_4080GeV_pPb_ExtGen_cfi',step1GenDefaults)
steps['Pyquen_ZeemumuJets_pt10_2760GeV_ExtGen']=genvalid('Pyquen_ZeemumuJets_pt10_2760GeV_ExtGen_cfi',step1GenDefaults)

# Generator Hadronization (Hadronization of LHE)
steps['WJetsLNu_13TeV_madgraph-pythia8']=genvalid('Hadronizer_MgmMatchTuneCUETP8M1_13TeV_madgraph_pythia8_cff',step1GenDefaults,dataSet='/WJetsToLNu_13TeV-madgraph/Fall13wmLHE-START62_V1-v1/GEN')
steps['Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCP5_13TeV_MLM_5f_max4j_LHE_pythia8']=genvalid('Hadronizer_TuneCP5_13TeV_MLM_5f_max4j_LHE_pythia8_cff',step1HadronizerDefaults)
steps['GGToH_13TeV_pythia8']=genvalid('GGToHtautau_13TeV_pythia8_cff',step1GenDefaults)

steps['WJetsLNutaupinu_13TeV_madgraph-pythia8']=genvalid('Hadronizer_MgmMatchTuneCUETP8M1_13TeV_madgraph_pythia8_taupinu_cff',step1GenDefaults,dataSet='/WJetsToLNu_13TeV-madgraph/Fall13wmLHE-START62_V1-v1/GEN')
steps['Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_taupinu']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_taupinu_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCP5_13TeV_MLM_5f_max4j_LHE_pythia8_taupinu']=genvalid('Hadronizer_TuneCP5_13TeV_MLM_5f_max4j_LHE_pythia8_taupinu_cff',step1HadronizerDefaults)
steps['GGToHtaupinu_13TeV_pythia8']=genvalid('GGToHtautau_13TeV_pythia8_taupinu_cff',step1GenDefaults)

steps['WJetsLNutaurhonu_13TeV_madgraph-pythia8']=genvalid('Hadronizer_MgmMatchTuneCUETP8M1_13TeV_madgraph_pythia8_taurhonu_cff.py',step1GenDefaults,dataSet='/WJetsToLNu_13TeV-madgraph/Fall13wmLHE-START62_V1-v1/GEN')
steps['Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_taurhonu']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_taurhonu_cff.py',step1HadronizerDefaults)
steps['Hadronizer_TuneCP5_13TeV_MLM_5f_max4j_LHE_pythia8_taurhonu']=genvalid('Hadronizer_TuneCP5_13TeV_MLM_5f_max4j_LHE_pythia8_taurhonu_cff.py',step1HadronizerDefaults)
steps['GGToHtaurhonu_13TeV_pythia8']=genvalid('GGToHtautau_13TeV_pythia8_taurhonu_cff',step1GenDefaults)

steps['Hadronizer_TuneCUETP8M1_13TeV_aMCatNLO_FXFX_5f_max2j_max1p_LHE_pythia8']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_aMCatNLO_FXFX_5f_max2j_max1p_LHE_pythia8_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCP5_13TeV_aMCatNLO_FXFX_5f_max2j_max1p_LHE_pythia8']=genvalid('Hadronizer_TuneCP5_13TeV_aMCatNLO_FXFX_5f_max2j_max1p_LHE_pythia8_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCUETP8M1_13TeV_aMCatNLO_FXFX_5f_max2j_max1p_LHE_pythia8_evtgen']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_aMCatNLO_FXFX_5f_max2j_max1p_LHE_pythia8_evtgen_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCP5_13TeV_aMCatNLO_FXFX_5f_max2j_max1p_LHE_pythia8_evtgen']=genvalid('Hadronizer_TuneCP5_13TeV_aMCatNLO_FXFX_5f_max2j_max1p_LHE_pythia8_evtgen_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCUETP8M1_13TeV_aMCatNLO_FXFX_5f_max2j_max0p_LHE_pythia8']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_aMCatNLO_FXFX_5f_max2j_max0p_LHE_pythia8_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCP5_13TeV_aMCatNLO_FXFX_5f_max2j_max0p_LHE_pythia8']=genvalid('Hadronizer_TuneCP5_13TeV_aMCatNLO_FXFX_5f_max2j_max0p_LHE_pythia8_cff',step1HadronizerDefaults)

steps['Hadronizer_TuneCUETP8M1_13TeV_Hgg_powhegEmissionVeto_pythia8']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_Hgg_powhegEmissionVeto_pythia8_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCP5_13TeV_Hgg_powhegEmissionVeto_pythia8']=genvalid('Hadronizer_TuneCP5_13TeV_Hgg_powhegEmissionVeto_pythia8_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCUETP8M1_13TeV_Httpinu_powhegEmissionVeto_pythia8']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_Httpinu_powhegEmissionVeto_pythia8_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCP5_13TeV_Httpinu_powhegEmissionVeto_pythia8']=genvalid('Hadronizer_TuneCP5_13TeV_Httpinu_powhegEmissionVeto_pythia8_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCUETP8M1_13TeV_Httrhonu_powhegEmissionVeto_pythia8']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_Httrhonu_powhegEmissionVeto_pythia8_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCP5_13TeV_Httrhonu_powhegEmissionVeto_pythia8']=genvalid('Hadronizer_TuneCP5_13TeV_Httrhonu_powhegEmissionVeto_pythia8_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCUETP8M1_13TeV_Htt_powhegEmissionVeto_pythia8']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_Htt_powhegEmissionVeto_pythia8_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCP5_13TeV_Htt_powhegEmissionVeto_pythia8']=genvalid('Hadronizer_TuneCP5_13TeV_Htt_powhegEmissionVeto_pythia8_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCUETP8M1_13TeV_Htt_powhegEmissionVeto_pythia8_tauola']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_Htt_powhegEmissionVeto_pythia8_tauola_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCP5_13TeV_Htt_powhegEmissionVeto_pythia8_tauola']=genvalid('Hadronizer_TuneCP5_13TeV_Htt_powhegEmissionVeto_pythia8_tauola_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCUETP8M1_13TeV_Httpinu_powhegEmissionVeto_pythia8_tauola']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_Httpinu_powhegEmissionVeto_pythia8_tauola_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCP5_13TeV_Httpinu_powhegEmissionVeto_pythia8_tauola']=genvalid('Hadronizer_TuneCP5_13TeV_Httpinu_powhegEmissionVeto_pythia8_tauola_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCUETP8M1_13TeV_Httrhonu_powhegEmissionVeto_pythia8_tauola']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_Httrhonu_powhegEmissionVeto_pythia8_tauola_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCP5_13TeV_Httrhonu_powhegEmissionVeto_pythia8_tauola']=genvalid('Hadronizer_TuneCP5_13TeV_Httrhonu_powhegEmissionVeto_pythia8_tauola_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCUETP8M1_13TeV_powhegEmissionVeto_pythia8']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_powhegEmissionVeto_pythia8_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCP5_13TeV_powhegEmissionVeto_pythia8']=genvalid('Hadronizer_TuneCP5_13TeV_powhegEmissionVeto_pythia8_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCUETP8M1_13TeV_powhegEmissionVeto2p_pythia8']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_powhegEmissionVeto2p_pythia8_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCP5_13TeV_powhegEmissionVeto2p_pythia8']=genvalid('Hadronizer_TuneCP5_13TeV_powhegEmissionVeto2p_pythia8_cff',step1HadronizerDefaults)

steps['Hadronizer_TuneCUETP8M1_Mad_pythia8']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_generic_LHE_pythia8_cff',step1HadronizerDefaults)



# Generator External Decays
steps['TT_13TeV_pythia8-evtgen']=genvalid('Hadronizer_MgmMatchTuneCUETP8M1_13TeV_madgraph_pythia8_EvtGen_cff',step1GenDefaults,dataSet='/TTJets_MSDecaysCKM_central_13TeV-madgraph/Fall13wmLHE-START62_V1-v1/GEN')

steps['Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCP5_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola']=genvalid('Hadronizer_TuneCP5_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola_cff',step1HadronizerDefaults)

steps['WToLNu_13TeV_pythia8-tauola']=genvalid('Hadronizer_MgmMatchTuneCUETP8M1_13TeV_madgraph_pythia8_Tauola_cff',step1GenDefaults,dataSet='/WJetsToLNu_13TeV-madgraph/Fall13wmLHE-START62_V1-v1/GEN')
steps['GGToH_13TeV_pythia8-tauola']=genvalid('GGToHtautau_13TeV_pythia8_Tauola_cff',step1GenDefaults)

steps['WToLNutaupinu_13TeV_pythia8-tauola']=genvalid('Hadronizer_MgmMatchTuneCUETP8M1_13TeV_madgraph_pythia8_Tauola_taupinu_cff',step1GenDefaults,dataSet='/WJetsToLNu_13TeV-madgraph/Fall13wmLHE-START62_V1-v1/GEN')
steps['Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola_taupinu']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola_taupinu_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCP5_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola_taupinu']=genvalid('Hadronizer_TuneCP5_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola_taupinu_cff',step1HadronizerDefaults)
steps['GGToHtaupinu_13TeV_pythia8-tauola']=genvalid('GGToHtautau_13TeV_pythia8_Tauola_taupinu_cff',step1GenDefaults)

steps['WToLNutaurhonu_13TeV_pythia8-tauola']=genvalid('Hadronizer_MgmMatchTuneCUETP8M1_13TeV_madgraph_pythia8_Tauola_taurhonu_cff',step1GenDefaults,dataSet='/WJetsToLNu_13TeV-madgraph/Fall13wmLHE-START62_V1-v1/GEN')
steps['Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola_taurhonu']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola_taurhonu_cff',step1HadronizerDefaults)
steps['Hadronizer_TuneCP5_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola_taurhonu']=genvalid('Hadronizer_TuneCP5_13TeV_MLM_5f_max4j_LHE_pythia8_Tauola_taurhonu_cff',step1HadronizerDefaults)
steps['GGToHtaurhonu_13TeV_pythia8-tauola']=genvalid('GGToHtautau_13TeV_pythia8_Tauola_taurhonu_cff',step1GenDefaults)

# normal fullSim workflows using gridpack LHE generator
# LHE-GEN-SIM step
step1LHEGenSimDefault = { '--relval':'9000,50',
                          '-s':'LHE,GEN,SIM',
                          '-n'            : 10,
                          '--conditions'  : 'auto:run2_mc',
                          '--beamspot'    : 'Realistic25ns13TeV2016Collision',
                          '--datatier'    : 'GEN-SIM,LHE',
                          '--eventcontent': 'FEVTDEBUG,LHE',
                          '--era'         : 'Run2_2016',
                        }

# LHE-GEN with DQM
step1LHEGenDQM = merge([{'-s':'LHE,GEN,VALIDATION:genvalid','--datatier'    : 'LHE,GEN,DQMIO','--eventcontent': 'LHE,RAWSIM,DQM'},step1LHEDefaults])


def lhegensim(fragment,howMuch):
    global step1LHEGenSimDefault
    return merge([{'cfg':fragment},howMuch,step1LHEGenSimDefault])

# LHE-GEN-SIM step for 2017
step1LHEGenSimUp2017Default = merge ([{'--conditions':'auto:phase1_2017_realistic','--era':'Run2_2017','--beamspot':'Realistic25ns13TeVEarly2017Collision','--geometry':'DB:Extended'},step1LHEGenSimDefault])

def lhegensim2017(fragment,howMuch):
    global step1LHEGenSimUp2017Default
    return merge([{'cfg':fragment},howMuch,step1LHEGenSimUp2017Default])

# LHE-GEN-SIM step for 2018
step1LHEGenSimUp2018Default = merge ([{'--conditions':'auto:phase1_2018_realistic','--era':'Run2_2018','--beamspot':'Realistic25ns13TeVEarly2018Collision','--geometry':'DB:Extended'},step1LHEGenSimDefault])

def lhegensim2018(fragment,howMuch):
    global step1LHEGenSimUp2018Default
    return merge([{'cfg':fragment},howMuch,step1LHEGenSimUp2018Default])

# Run-Dependent MC
kevents_rd = 9
events_per_job_rd = 50
def gen2018RD(fragment,howMuch):
    global step1Up2018Defaults
    return merge([{'cfg':fragment},howMuch,{'--conditions':'auto:phase1_2018_realistic_rd','--customise_commands': "\"process.source.numberEventsInLuminosityBlock=cms.untracked.uint32(" + str(events_per_job_rd) + ") \""},step1Up2018Defaults])

steps['ZEE_13UP18_RD']=gen2018RD('ZEE_13TeV_TuneCUETP8M1_cfi',Kby(kevents_rd,events_per_job_rd))
steps['ZMM_13UP18_RD']=gen2018RD('ZMM_13TeV_TuneCUETP8M1_cfi',Kby(kevents_rd,events_per_job_rd))
steps['TTbar_13UP18_RD']=gen2018RD('TTbar_13TeV_TuneCUETP8M1_cfi',Kby(kevents_rd,events_per_job_rd))
steps['TTbar_13UP18_RD_IB']=gen2018RD('TTbar_13TeV_TuneCUETP8M1_cfi',Kby(kevents_rd,events_per_job_rd))
steps['TTbar_13UP18_RD_IB']['--customise_commands'] = "\"process.source.numberEventsInLuminosityBlock=cms.untracked.uint32(5)\""

steps['TTbar012Jets_NLO_Mad_py8_Evt_13']=lhegensim('Configuration/Generator/python/TTbar012Jets_5f_NLO_FXFX_Madgraph_LHE_13TeV_cfi.py',Kby(9,50))
steps['GluGluHToZZTo4L_M125_Pow_py8_Evt_13']=lhegensim('Configuration/Generator/python/GGHZZ4L_JHUGen_Pow_NNPDF30_LHE_13TeV_cfi.py', Kby(9,50))
steps['VBFHToZZTo4Nu_M125_Pow_py8_Evt_13']=lhegensim('Configuration/Generator/python/VBFHZZ4Nu_Pow_NNPDF30_LHE_13TeV_cfi.py',Kby(9,50))
steps['VBFHToBB_M125_Pow_py8_Evt_13']=lhegensim('Configuration/Generator/python/VBFHbb_Pow_NNPDF30_LHE_13TeV_cfi.py',Kby(9,50))

steps['GluGluHToZZTo4L_M125_Pow_py8_Evt_13UP17']=lhegensim2017('Configuration/Generator/python/GGHZZ4L_JHUGen_Pow_NNPDF30_LHE_13TeV_cfi.py', Kby(9,100))
steps['VBFHToZZTo4Nu_M125_Pow_py8_Evt_13UP17']=lhegensim2017('Configuration/Generator/python/VBFHZZ4Nu_Pow_NNPDF30_LHE_13TeV_cfi.py',Kby(9,100))
steps['VBFHToBB_M125_Pow_py8_Evt_13UP17']=lhegensim2017('Configuration/Generator/python/VBFHbb_Pow_NNPDF30_LHE_13TeV_cfi.py',Kby(9,100))

steps['GluGluHToZZTo4L_M125_Pow_py8_Evt_13UP18']=lhegensim2018('Configuration/Generator/python/GGHZZ4L_JHUGen_Pow_NNPDF30_LHE_13TeV_cfi.py', Kby(9,100))
steps['VBFHToZZTo4Nu_M125_Pow_py8_Evt_13UP18']=lhegensim2018('Configuration/Generator/python/VBFHZZ4Nu_Pow_NNPDF30_LHE_13TeV_cfi.py',Kby(9,100))
steps['VBFHToBB_M125_Pow_py8_Evt_13UP18']=lhegensim2018('Configuration/Generator/python/VBFHbb_Pow_NNPDF30_LHE_13TeV_cfi.py',Kby(9,100))

#GEN-SIM inputs for LHE-GEN-SIM workflows
#steps['TTbar012Jets_NLO_Mad_py8_Evt_13INPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar012Jets_NLO_Mad_py8_Evt_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
#steps['GluGluHToZZTo4L_M125_Pow_py8_Evt_13INPUT']={'INPUT':InputInfo(dataSet='/RelValGluGluHToZZTo4L_M125_Pow_py8_Evt_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
#steps['VBFHToZZTo4Nu_M125_Pow_py8_Evt_13INPUT']={'INPUT':InputInfo(dataSet='/RelValVBFHToZZTo4Nu_M125_Pow_py8_Evt_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
#steps['VBFHToBB_M125_Pow_py8_Evt_13INPUT']={'INPUT':InputInfo(dataSet='/RelValVBFHToBB_M125_Pow_py8_Evt_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
#steps['VBFHToZZTo4Nu_M125_Pow_py8_Evt_13UP17INPUT']={'INPUT':InputInfo(dataSet='/RelValVBFHToZZTo4Nu_M125_Pow_py8_Evt_13UP17/%s/GEN-SIM'%(baseDataSetRelease[21],),location='STD')}
#steps['VBFHToBB_M125_Pow_py8_Evt_13UP17INPUT']={'INPUT':InputInfo(dataSet='/RelValVBFHToBB_M125_Pow_py8_Evt_13UP17/%s/GEN-SIM'%(baseDataSetRelease[21],),location='STD')}
#steps['VBFHToZZTo4Nu_M125_Pow_py8_Evt_13UP18INPUT']={'INPUT':InputInfo(dataSet='/RelValVBFHToZZTo4Nu_M125_Pow_py8_Evt_13UP18/%s/GEN-SIM'%(baseDataSetRelease[22],),location='STD')}
#steps['VBFHToBB_M125_Pow_py8_Evt_13UP18INPUT']={'INPUT':InputInfo(dataSet='/RelValVBFHToBB_M125_Pow_py8_Evt_13UP18/%s/GEN-SIM'%(baseDataSetRelease[22],),location='STD')}

# normal fullSim workflows using pLHE-GEN-SIM 2016 by default
# pLHE step
step1LHENormal = {'--relval'     : '10000,50',
                 '--mc'          : '',
                 '-s'            : 'NONE',
                 '--conditions'  : 'auto:run2_mc',
                 '--datatier'    : 'LHE',
                 '--eventcontent': 'LHE',
                 '--era'         : 'Run2_2016',
                 '-n'            : 10,
                 }
# followed by GEN-SIM step
step1GENNormal = {'--relval'     : '10000,50',
                 '-s'           : 'GEN,SIM',
                 '--conditions'  : 'auto:run2_mc',
                 '--beamspot'    : 'Realistic25ns13TeV2016Collision',
                 '--eventcontent': 'FEVTDEBUG',
                 '--datatier'    : 'GEN-SIM',
                 '--era'         : 'Run2_2016',
                 }

# pLHE-GEN-SIM step for 2017 conditions
step1LHENormal2017Default = merge ([{'--conditions':'auto:phase1_2017_realistic','--era':'Run2_2017','--beamspot':'Realistic25ns13TeVEarly2017Collision','--geometry':'DB:Extended'},step1LHENormal])
step1GENNormal2017Default = merge ([{'--conditions':'auto:phase1_2017_realistic','--era':'Run2_2017','--beamspot':'Realistic25ns13TeVEarly2017Collision','--geometry':'DB:Extended'},step1GENNormal])

# pLHE-GEN-SIM step for 2018 conditions
step1LHENormal2018Default = merge ([{'--conditions':'auto:phase1_2018_realistic','--era':'Run2_2018','--beamspot':'Realistic25ns13TeVEarly2018Collision','--geometry':'DB:Extended'},step1LHENormal])
step1GENNormal2018Default = merge ([{'--conditions':'auto:phase1_2018_realistic','--era':'Run2_2018','--beamspot':'Realistic25ns13TeVEarly2018Collision','--geometry':'DB:Extended'},step1GENNormal])

# pLHE and GEN-SIM steps 2016 workflows
steps['GluGluHToGG_M125_Pow_MINLO_NNLOPS_py8_13']=merge ([{'--filein':'lhe:18334'},step1LHENormal])
steps['Hadronizer_TuneCUETP8M1_13TeV_powhegEmissionVeto_2p_HToGG_M125_13']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_powhegEmissionVeto_2p_HToGG_M125_LHE_py8_cff',step1GENNormal)

# pLHE and GEN-SIM steps 2017 workflows
steps['GluGluHToGG_M125_Pow_MINLO_NNLOPS_py8_13UP17']=merge ([{'--filein':'lhe:18334'},step1LHENormal2017Default])
steps['Hadronizer_TuneCUETP8M1_13TeV_powhegEmissionVeto_2p_HToGG_M125_13UP17']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_powhegEmissionVeto_2p_HToGG_M125_LHE_py8_cff',step1GENNormal2017Default)

# pLHE and GEN-SIM steps 2018 workflows
steps['GluGluHToGG_M125_Pow_MINLO_NNLOPS_py8_13UP18']=merge ([{'--filein':'lhe:18334'},step1LHENormal2018Default])
steps['Hadronizer_TuneCUETP8M1_13TeV_powhegEmissionVeto_2p_HToGG_M125_13UP18']=genvalid('Hadronizer_TuneCUETP8M1_13TeV_powhegEmissionVeto_2p_HToGG_M125_LHE_py8_cff',step1GENNormal2018Default)

#GEN-SIM inputs for pLHE-GEN-SIM workflows 2016,2017,2018
#steps['GluGluHToGG_M125_Pow_MINLO_NNLOPS_py8_13INPUT']={'INPUT':InputInfo(dataSet='/RelValGluGluHToGG_M125_Pow_MINLO_NNLOPS_py8_13/%s/GEN-SIM'%(baseDataSetRelease[3],),location='STD')}
#steps['GluGluHToGG_M125_Pow_MINLO_NNLOPS_py8_13UP17INPUT']={'INPUT':InputInfo(dataSet='/RelValGluGluHToGG_M125_Pow_MINLO_NNLOPS_py8_13UP17/%s/GEN-SIM'%(baseDataSetRelease[21],),location='STD')}
#steps['GluGluHToGG_M125_Pow_MINLO_NNLOPS_py8_13UP18INPUT']={'INPUT':InputInfo(dataSet='/RelValGluGluHToGG_M125_Pow_MINLO_NNLOPS_py8_13UP18/%s/GEN-SIM'%(baseDataSetRelease[22],),location='STD')}


#Sherpa
steps['sherpa_ZtoEE_0j_BlackHat_13TeV_MASTER']=genvalid('sherpa_ZtoEE_0j_BlackHat_13TeV_MASTER_cff',step1GenDefaults)
steps['sherpa_ZtoLL_2j_MEPSatNLO_13TeV_MASTER']=genvalid('sherpa_ZtoLL_2j_MEPSatNLO_13TeV_MASTER_cff',step1GenDefaults)
steps['sherpa_ttbar_2j_MENLOPS_13TeV_MASTER']=genvalid('sherpa_ttbar_2j_MENLOPS_13TeV_MASTER_cff',step1GenDefaults)

#Herwig7
steps['TTbar_13TeV_Pow_herwig7']=genvalid('Configuration/Generator/python/TT_13TeV_Pow_Herwig7_cff',step1LHEGenDQM)
steps['DYToLL012Jets_5FS_TuneCH3_13TeV_amcatnloFxFx_herwig7']=genvalid('Configuration/Generator/python/DYToLL012Jets_5FS_TuneCH3_13TeV_amcatnloFxFx_herwig7_cff',merge([{'-n':'12'},step1LHEGenDQM]))
steps['DYToLL01234Jets_5FS_TuneCH3_13TeV_madgraphMLM_herwig7']=genvalid('Configuration/Generator/python/DYToLL01234Jets_5FS_TuneCH3_13TeV_madgraphMLM_herwig7_cff',step1LHEGenDQM)


# Heavy Ion
steps['ReggeGribovPartonMC_EposLHC_5TeV_pPb']=genvalid('GeneratorInterface/ReggeGribovPartonMCInterface/ReggeGribovPartonMC_EposLHC_5TeV_pPb_cfi',step1GenDefaults)

# B-physics
steps['BuToKstarJPsiToMuMu_forSTEAM_13TeV']=genvalid('BuToKstarJPsiToMuMu_forSTEAM_13TeV_cfi',step1GenDefaults)
steps['Upsilon4swithBuToKstarJPsiToMuMu_forSTEAM_13TeV_TuneCUETP8M1']=genvalid('Upsilon4swithBuToKstarJPsiToMuMu_forSTEAM_13TeV_TuneCUETP8M1_cfi',step1GenDefaults)
steps['Upsilon4sBaBarExample_BpBm_Dstarpipi_D0Kpi_nonres_forSTEAM_13TeV_TuneCUETP8M1']=genvalid('Upsilon4sBaBarExample_BpBm_Dstarpipi_D0Kpi_nonres_forSTEAM_13TeV_TuneCUETP8M1_cfi',step1GenDefaults)
steps['LambdaBToLambdaMuMuToPPiMuMu_forSTEAM_13TeV_TuneCUETP8M1']=genvalid('LambdaBToLambdaMuMuToPPiMuMu_forSTEAM_13TeV_TuneCUETP8M1_cfi',step1GenDefaults)
steps['BsToMuMu_forSTEAM_13TeV']=genvalid('BsToMuMu_forSTEAM_13TeV_cfi',step1GenDefaults)

# Workflows for multiple concurrent lumi blocks

def lhegensim2018ml(fragment,howMuch):
    return merge([{'cfg':fragment},howMuch,{'--customise_commands': '"process.source.numberEventsInLuminosityBlock=cms.untracked.uint32(5)"'},step1LHEGenSimUp2018Default])

steps['GluGluHToZZTo4L_M125_Pow_py8_Evt_13UP18ml']=lhegensim2018ml('Configuration/Generator/python/GGHZZ4L_JHUGen_Pow_NNPDF30_LHE_13TeV_cfi.py',Kby(9,50))



# sometimes v1 won't be used - override it here - the dictionary key is gen fragment + '_' + geometry
overrideFragments={'H125GGgluonfusion_13UP18INPUT':'2'}

import re
for key in overrideFragments:
    for inI in steps[key]:
        DSold=steps[key][inI].dataSet
        DS = re.sub('v[0-9]*/','v'+overrideFragments[key]+'/',DSold.rstrip())
        del steps[key]
        steps[key]={'INPUT':InputInfo(dataSet=DS,location='STD')}


#PU for FullSim
PU={'-n':10,'--pileup':'default','--pileup_input':'das:/RelValMinBias/%s/GEN-SIM'%(baseDataSetRelease[0],)}
# pu2 can be removed
PU2={'-n':10,'--pileup':'default','--pileup_input':'das:/RelValMinBias/%s/GEN-SIM'%(baseDataSetRelease[0],)}
PU25={'-n':10,'--pileup':'AVE_35_BX_25ns','--pileup_input':'das:/RelValMinBias_13/%s/GEN-SIM'%(baseDataSetRelease[3],)}
PU50={'-n':10,'--pileup':'AVE_35_BX_50ns','--pileup_input':'das:/RelValMinBias_13/%s/GEN-SIM'%(baseDataSetRelease[3],)}
PUHI={'-n':10,'--pileup_input':'das:/RelValHydjetQ_B12_5020GeV_2018/%s/GEN-SIM'%(baseDataSetRelease[9])}
PUHI2022={'-n':10,'--pileup_input':'das:/RelValHydjetQ_B12_5020GeV_2021_ppReco/%s/GEN-SIM'%(baseDataSetRelease[23])}
PUHI2023={'-n':10,'--pileup_input':'das:/RelValHydjetQ_MinBias_5362GeV_2023_ppReco/%s/GEN-SIM'%(baseDataSetRelease[25])}
PU25UP17={'-n':10,'--pileup':'AVE_35_BX_25ns','--pileup_input':'das:/RelValMinBias_13/%s/GEN-SIM'%(baseDataSetRelease[13],)}
PU25UP18={'-n':10,'--pileup':'AVE_50_BX_25ns','--pileup_input':'das:/RelValMinBias_13/%s/GEN-SIM'%(baseDataSetRelease[18],)}

#PU for FastSim
# FS_PU_INPUT_13TEV = "file:/afs/cern.ch/work/l/lveldere/minbias.root" # placeholder for relval to be produced with wf  135.8
PUFS={'--pileup':'GEN_2012_Summer_50ns_PoissonOOTPU'}
# PUFS2={'--pileup':'2012_Startup_50ns_PoissonOOTPU'} # not used anywhere
PUFSAVE10={'--pileup':'GEN_AVE_10_BX_25ns'}  # temporary: one or a few releases as back-up
PUFSAVE20={'--pileup':'GEN_AVE_20_BX_25ns'}  # temporary: one or a few releases as back-up
PUFSAVE35={'--pileup':'GEN_AVE_35_BX_25ns'}
PUFSAVE10_DRMIX_ITO={'--pileup':'AVE_10_BX_25ns','--pileup_input':'das:/RelValMinBiasFS_13_ForMixing/%s/GEN-SIM-RECO'%(baseDataSetRelease[7],),'--era':'Run2_25ns','--customise':'FastSimulation/Configuration/Customs.disableOOTPU'}
PUFSAVE35_DRMIX_ITO={'--pileup':'AVE_35_BX_25ns','--pileup_input':'das:/RelValMinBiasFS_13_ForMixing/%s/GEN-SIM-RECO'%(baseDataSetRelease[7],),'--era':'Run2_25ns','--customise':'FastSimulation/Configuration/Customs.disableOOTPU'}
PUFS25={'--pileup':'AVE_35_BX_25ns','--pileup_input':'das:/RelValMinBiasFS_13_ForMixing/%s/GEN-SIM-RECO'%(baseDataSetRelease[7],)}

#pu25 for high stats workflows
PU25HS={'-n':10,'--pileup':'AVE_35_BX_25ns','--pileup_input':'das:/RelValMinBias_13/%s/GEN-SIM'%(baseDataSetRelease[11],)}


#
steps['TTbarFSPU']=merge([PUFS,Kby(100,500),steps['TTbarFS']] )

steps['FS_TTbar_13_PUAVE10']=merge([PUFSAVE10,Kby(100,500),steps['TTbarFS_13']] ) # temporary: one or a few releases as back-up
steps['FS_TTbar_13_PUAVE20']=merge([PUFSAVE20,Kby(100,500),steps['TTbarFS_13']] ) # temporary: one or a few releases as back-up
steps['FS_TTbar_13_PUAVE35']=merge([PUFSAVE35,Kby(100,500),steps['TTbarFS_13']] )
steps['FS_TTbar_13_PU25']=merge([PUFS25,Kby(100,500),steps['TTbarFS_13']] ) # needs the placeholder
steps['FS_NuGun_UP15_PU25']=merge([PUFS25,Kby(100,500),steps['NuGunFS_UP15']] ) # needs the placeholder
steps['FS_SMS-T1tttt_mGl-1500_mLSP-100_13_PU25']=merge([PUFS25,Kby(100,500),steps['SMS-T1tttt_mGl-1500_mLSP-100FS_13']] )

steps['FS__PU25']=merge([PUFS25,Kby(100,500),steps['NuGunFS_UP15']] ) # needs the placeholder
steps['FS_TTbar_13_PUAVE10_DRMIX_ITO']=merge([PUFSAVE10_DRMIX_ITO,Kby(100,500),steps['TTbarFS_13']] ) # needs the placeholder
steps['FS_TTbar_13_PUAVE35_DRMIX_ITO']=merge([PUFSAVE35_DRMIX_ITO,Kby(100,500),steps['TTbarFS_13']] ) # needs the placeholder

# step2
step2Defaults = { '-s'            : 'DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@fake',
                  '--datatier'    : 'GEN-SIM-DIGI-RAW-HLTDEBUG',
                  '--eventcontent': 'FEVTDEBUGHLT',
                  '--conditions'  : 'auto:run1_mc',
                  }
#for 2015
step2Upg2015Defaults = {'-s'     :'DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval2016',
                 '--conditions'  :'auto:run2_mc',
                 '--datatier'    :'GEN-SIM-DIGI-RAW-HLTDEBUG',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--era'         :'Run2_2016',
                 '-n'            :'10',
                  }
step2Upg2015Defaults50ns = merge([{'-s':'DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval50ns','--conditions':'auto:run2_mc_50ns','--era':'Run2_50ns'},step2Upg2015Defaults])

step2Upg2015DefaultsAPVSimu = merge([{'--era': 'Run2_2016_HIPM'},step2Upg2015Defaults])

#for 2017
step2Upg2017Defaults = {'-s'     :'DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval2017',
                 '--conditions'  :'auto:phase1_2017_realistic',
                 '--datatier'    :'GEN-SIM-DIGI-RAW-HLTDEBUG',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--era'         :'Run2_2017',
                 '-n'            :'10',
                 '--geometry'    :'DB:Extended',
                  }

#for 2018
step2Upg2018Defaults = {'-s'     :'DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval2018',
                 '--conditions'  :'auto:phase1_2018_realistic',
                 '--datatier'    :'GEN-SIM-DIGI-RAW-HLTDEBUG',
                 '--eventcontent':'FEVTDEBUGHLT',
                 '--era'         :'Run2_2018',
                 '-n'            :'10',
                 '--geometry'    :'DB:Extended',
                  }


steps['DIGIUP15']=merge([step2Upg2015Defaults])
steps['DIGIUP15APVSimu']=merge([{'--era': 'Run2_2016_HIPM'},step2Upg2015Defaults])

steps['DIGIUP15PROD1']=merge([{'-s':'DIGI,L1,DIGI2RAW,HLT:@relval2016','--eventcontent':'RAWSIM','--datatier':'GEN-SIM-RAW'},step2Upg2015Defaults])
steps['DIGIUP15_PU25']=merge([PU25,step2Upg2015Defaults])
steps['DIGIUP15APVSimu_PU25']=merge([PU25,step2Upg2015DefaultsAPVSimu])

steps['DIGIUP15_PU50']=merge([PU50,step2Upg2015Defaults50ns])
steps['DIGIUP17']=merge([step2Upg2017Defaults])
steps['DIGIUP18']=merge([step2Upg2018Defaults])
steps['DIGIUP17PROD1']=merge([{'-s':'DIGI,L1,DIGI2RAW,HLT:@relval2017','--eventcontent':'RAWSIM','--datatier':'GEN-SIM-RAW'},step2Upg2017Defaults])
steps['DIGIUP18PROD1']=merge([{'-s':'DIGI,L1,DIGI2RAW,HLT:@relval2018','--eventcontent':'RAWSIM','--datatier':'GEN-SIM-RAW'},step2Upg2018Defaults])
steps['DIGIUP18PROD1bParking']=merge([{'-s':'DIGI,L1,DIGI2RAW,HLT:@relval2018','--eventcontent':'RAWSIM','--datatier':'GEN-SIM-RAW','--era' :'Run2_2018,bParking'},step2Upg2018Defaults])
steps['DIGIUP17_PU25']=merge([PU25UP17,step2Upg2017Defaults])
steps['DIGIUP18_PU25']=merge([PU25UP18,step2Upg2018Defaults])

# for Run2 PPb MC workflows
steps['DIGIUP15_PPb']=merge([{'-s':'DIGI:pdigi_valid,L1,DIGI2RAW,HLT:PIon','--conditions':'auto:run2_mc_pa', '--era':'Run2_2016_pA'}, steps['DIGIUP15']])

# PU25 for high stats workflows
steps['DIGIUP15_PU25HS']=merge([PU25HS,step2Upg2015Defaults])


steps['DIGIPROD1']=merge([{'-s':'DIGI,L1,DIGI2RAW,HLT:@fake','--eventcontent':'RAWSIM','--datatier':'GEN-SIM-RAW'},step2Defaults])
steps['DIGI']=merge([step2Defaults])
#steps['DIGI2']=merge([stCond,step2Defaults])
steps['DIGICOS']=merge([{'--scenario':'cosmics','--eventcontent':'FEVTDEBUG','--datatier':'GEN-SIM-DIGI-RAW'},stCond,step2Defaults])
steps['DIGIHAL']=merge([{'--scenario':'cosmics','--eventcontent':'FEVTDEBUG','--datatier':'GEN-SIM-DIGI-RAW'},step2Upg2015Defaults])
steps['DIGICOS_UP15']=merge([{'--conditions':'auto:run2_mc_cosmics','--scenario':'cosmics','--eventcontent':'FEVTDEBUG','--datatier':'GEN-SIM-DIGI-RAW'},step2Upg2015Defaults])
steps['DIGICOS_UP16']=merge([{'--conditions':'auto:run2_mc_cosmics','-s':'DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval2016','--scenario':'cosmics','--eventcontent':'FEVTDEBUG','--datatier':'GEN-SIM-DIGI-RAW', '--era' : 'Run2_2016'},step2Upg2015Defaults])
steps['DIGICOS_UP17']=merge([{'--conditions':'auto:phase1_2017_cosmics','-s':'DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval2017','--scenario':'cosmics','--eventcontent':'FEVTDEBUG','--datatier':'GEN-SIM-DIGI-RAW', '--era' : 'Run2_2017'},step2Upg2015Defaults])
steps['DIGICOS_UP18']=merge([{'--conditions':'auto:phase1_2018_cosmics','-s':'DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval2018','--scenario':'cosmics','--eventcontent':'FEVTDEBUG','--datatier':'GEN-SIM-DIGI-RAW', '--era' : 'Run2_2018'},step2Upg2015Defaults])
steps['DIGICOS_UP21']=merge([{'--conditions':'auto:phase1_2022_cosmics','-s':'DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval2017','--scenario':'cosmics','--eventcontent':'FEVTDEBUG','--datatier':'GEN-SIM-DIGI-RAW', '--era' : 'Run3'},step2Upg2015Defaults])
steps['DIGICOS_UP21_0T']=merge([{'--magField':'0T','--conditions':'auto:phase1_2022_cosmics_0T'},steps['DIGICOS_UP21']])

steps['DIGICOSPEAK_UP17']=merge([{'--conditions':'auto:phase1_2017_cosmics_peak','-s':'DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval2017','--customise_commands': '"process.mix.digitizers.strip.APVpeakmode=cms.bool(True)"','--scenario':'cosmics','--eventcontent':'FEVTDEBUG','--datatier':'GEN-SIM-DIGI-RAW', '--era' : 'Run2_2017'},step2Upg2015Defaults])
steps['DIGICOSPEAK_UP18']=merge([{'--conditions':'auto:phase1_2018_cosmics_peak','-s':'DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval2018','--customise_commands': '"process.mix.digitizers.strip.APVpeakmode=cms.bool(True)"','--scenario':'cosmics','--eventcontent':'FEVTDEBUG','--datatier':'GEN-SIM-DIGI-RAW', '--era' : 'Run2_2018'},step2Upg2015Defaults])

steps['DIGIPU1']=merge([PU,step2Defaults])
steps['DIGIPU2']=merge([PU2,step2Defaults])
steps['REDIGIPU']=merge([{'-s':'reGEN,reDIGI,L1,DIGI2RAW,HLT:@fake'},steps['DIGIPU1']])

steps['DIGIUP15_ID']=merge([{'--restoreRND':'HLT','--process':'HLT2'},steps['DIGIUP15']])

steps['RESIM']=merge([{'-s':'reGEN,reSIM','-n':10},steps['DIGI']])
#steps['RESIMDIGI']=merge([{'-s':'reGEN,reSIM,DIGI,L1,DIGI2RAW,HLT:@fake,RAW2DIGI,L1Reco','-n':10,'--restoreRNDSeeds':'','--process':'HLT'},steps['DIGI']])


steps['DIGIHI2023PPRECOAPPROXCLUSTERS']=merge([{'-s':'DIGI:pdigi_hi_nogen,L1,DIGI2RAW,HLT:@fake2'}, hiDefaults2023_ppReco_approxClusters, {'--pileup':'HiMixNoPU'}, step2Upg2015Defaults])
steps['DIGIHI2023PPRECO']=merge([{'-s':'DIGI:pdigi_hi_nogen,L1,DIGI2RAW,HLT:@fake2'}, hiDefaults2023_ppReco, {'--pileup':'HiMixNoPU'}, step2Upg2015Defaults])
steps['DIGIHI2022PPRECOAPPROXCLUSTERS']=merge([{'-s':'DIGI:pdigi_hi_nogen,L1,DIGI2RAW,HLT:@fake2'}, hiDefaults2022_ppReco_approxClusters, {'--pileup':'HiMixNoPU'}, step2Upg2015Defaults])
steps['DIGIHI2022PPRECO']=merge([{'-s':'DIGI:pdigi_hi_nogen,L1,DIGI2RAW,HLT:@fake2'}, hiDefaults2022_ppReco, {'--pileup':'HiMixNoPU'}, step2Upg2015Defaults])
steps['DIGIHI2018PPRECO']=merge([{'-s':'DIGI:pdigi_hi_nogen,L1,DIGI2RAW,HLT:@fake2'}, hiDefaults2018_ppReco, {'--pileup':'HiMixNoPU'}, step2Upg2015Defaults])
steps['DIGIHI2018']=merge([{'-s':'DIGI:pdigi_hi_nogen,L1,DIGI2RAW,HLT:@fake2'}, hiDefaults2018, {'--pileup':'HiMixNoPU'}, step2Upg2015Defaults])
steps['DIGIHI2017']=merge([{'-s':'DIGI:pdigi_hi_nogen,L1,DIGI2RAW,HLT:@fake2'}, hiDefaults2017, step2Upg2015Defaults])
steps['DIGIHI2015']=merge([{'-s':'DIGI:pdigi_hi_nogen,L1,DIGI2RAW,HLT:@fake'}, hiDefaults2015, {'--pileup':'HiMixNoPU'}, step2Upg2015Defaults])
steps['DIGIHI2011']=merge([{'-s':'DIGI:pdigi_hi_nogen,L1,DIGI2RAW,HLT:@fake'}, hiDefaults2011, {'--pileup':'HiMixNoPU'}, step2Defaults])
steps['DIGIHI2023MIX']=merge([{'-s':'DIGI:pdigi_hi_nogen,L1,DIGI2RAW,HLT:@fake2', '-n':2}, hiDefaults2023_ppReco, {'--pileup':'HiMix'}, PUHI2023, step2Upg2015Defaults])
steps['DIGIHI2022MIX']=merge([{'-s':'DIGI:pdigi_hi_nogen,L1,DIGI2RAW,HLT:@fake2', '-n':2}, hiDefaults2022_ppReco, {'--pileup':'HiMix'}, PUHI2022, step2Upg2015Defaults])
steps['DIGIHIMIX']=merge([{'-s':'DIGI:pdigi_hi_nogen,L1,DIGI2RAW,HLT:@fake2', '-n':2}, hiDefaults2018_ppReco, {'--pileup':'HiMix'}, PUHI, step2Upg2015Defaults])

steps['DIGIPPREF2017']=merge([{'-s':'DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@fake2'}, ppRefDefaults2017, step2Upg2015Defaults])

# PRE-MIXING : https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideSimulation#Pre_Mixing_Instructions
premixUp2015Defaults = {
    '--evt_type'    : 'SingleNuE10_cfi',
    '-s'            : 'GEN,SIM,DIGI:pdigi_valid',
    '-n'            : '10',
    '--conditions'  : 'auto:run2_mc', # 25ns GT; dedicated dict for 50ns
    '--datatier'    : 'PREMIX',
    '--eventcontent': 'PREMIX',
    '--procModifiers':'premix_stage1',
    '--era'         : 'Run2_2016' # temporary replacement for premix; to be brought back to customisePostLS1 *EDIT - This comment possibly no longer relevant with switch to eras
}

premixUp2015Defaults50ns = merge([{'--conditions':'auto:run2_mc_50ns'},
                                  {'--era':'Run2_50ns'},
                                  premixUp2015Defaults])

premixUp2017Defaults = merge([{'--conditions':'auto:phase1_2017_realistic'},
                                  {'--era':'Run2_2017'},
                                  premixUp2015Defaults])

premixUp2018Defaults = merge([{'--conditions':'auto:phase1_2018_realistic'},
                                  {'--era':'Run2_2018'},
                                  premixUp2015Defaults])

premixUp2018_RDPix = merge([{'--conditions':'auto:phase1_2018_realistic_rd'},
                                  {'--era':'Run2_2018'},
                                  {'--procModifiers':'premix_stage1,runDependentForPixel'},
                                  premixUp2015Defaults])

steps['PREMIXUP15_PU25']=merge([PU25,Kby(100,100),premixUp2015Defaults])
steps['PREMIXUP15_PU50']=merge([PU50,Kby(100,100),premixUp2015Defaults50ns])
steps['PREMIXUP17_PU25']=merge([PU25UP17,Kby(100,100),premixUp2017Defaults])
steps['PREMIXUP18_PU25']=merge([PU25UP18,Kby(100,100),premixUp2018Defaults])
steps['PREMIXUP18_PU25_RDPix']=merge([PU25UP18,Kby(100,100),premixUp2018_RDPix])

digiPremixUp2015Defaults25ns = {
    '--conditions'   : 'auto:run2_mc',
    '-s'             : 'DIGI:pdigi_valid,DATAMIX,L1,DIGI2RAW,HLT:@relval2016',
    '--pileup_input'  :  'das:/RelValPREMIXUP15_PU25/%s/PREMIX'%baseDataSetRelease[5],
    '--eventcontent' : 'FEVTDEBUGHLT',
    '--datatier'     : 'GEN-SIM-DIGI-RAW-HLTDEBUG',
    '--datamix'      : 'PreMix',
    '--procModifiers': 'premix_stage2',
    '--era'          : 'Run2_2016'
    }

from .upgradeWorkflowComponents import digiPremixLocalPileup
digiPremixLocalPileupUp2015Defaults25ns = merge([digiPremixLocalPileup,
                                                 digiPremixUp2015Defaults25ns])
digiPremixUp2015Defaults50ns=merge([{'-s':'DIGI:pdigi_valid,DATAMIX,L1,DIGI2RAW,HLT:@relval50ns'},
                                    {'--conditions':'auto:run2_mc_50ns'},
                                    {'--pileup_input' : 'das:/RelValPREMIXUP15_PU50/%s/PREMIX'%baseDataSetRelease[6]},
                                    {'--era'            : 'Run2_50ns'},
                                    digiPremixUp2015Defaults25ns])

digiPremixUp2017Defaults25ns = {
    '--conditions'   : 'auto:phase1_2017_realistic',
    '-s'             : 'DIGI:pdigi_valid,DATAMIX,L1,DIGI2RAW,HLT:@relval2017',
    '--pileup_input'  :  'das:/RelValPREMIXUP17_PU25/%s/PREMIX'%baseDataSetRelease[14],
    '--eventcontent' : 'FEVTDEBUGHLT',
    '--datatier'     : 'GEN-SIM-DIGI-RAW-HLTDEBUG',
    '--datamix'      : 'PreMix',
    '--procModifiers': 'premix_stage2',
    '--era'          : 'Run2_2017'
    }
digiPremixLocalPileupUp2017Defaults25ns = merge([digiPremixLocalPileup,
                                                 digiPremixUp2017Defaults25ns])


digiPremixUp2018Defaults25ns = {
    '--conditions'   : 'auto:phase1_2018_realistic',
    '-s'             : 'DIGI:pdigi_valid,DATAMIX,L1,DIGI2RAW,HLT:@relval2018',
    '--pileup_input'  :  'das:/RelValPREMIXUP18_PU25/%s/PREMIX'%baseDataSetRelease[17],
    '--eventcontent' : 'FEVTDEBUGHLT',
    '--datatier'     : 'GEN-SIM-DIGI-RAW-HLTDEBUG',
    '--datamix'      : 'PreMix',
    '--procModifiers': 'premix_stage2',
    '--era'          : 'Run2_2018'
    }
digiPremixLocalPileupUp2018Defaults25ns = merge([digiPremixLocalPileup,
                                                 digiPremixUp2018Defaults25ns])

steps['DIGIPRMXUP15_PU25']=merge([digiPremixUp2015Defaults25ns])
steps['DIGIPRMXUP15APVSimu_PU25']=merge([{'--era': 'Run2_2016_HIPM'},digiPremixUp2015Defaults25ns])
steps['DIGIPRMXLOCALUP15_PU25']=merge([digiPremixLocalPileupUp2015Defaults25ns])
steps['DIGIPRMXLOCALUP15APVSimu_PU25']=merge([{'--era': 'Run2_2016_HIPM'},digiPremixLocalPileupUp2015Defaults25ns])

steps['DIGIPRMXUP15_PU50']=merge([digiPremixUp2015Defaults50ns])
steps['DIGIPRMXUP17_PU25']=merge([digiPremixUp2017Defaults25ns])
steps['DIGIPRMXLOCALUP17_PU25']=merge([digiPremixLocalPileupUp2017Defaults25ns])
steps['DIGIPRMXUP18_PU25']=merge([digiPremixUp2018Defaults25ns])
steps['DIGIPRMXLOCALUP18_PU25']=merge([digiPremixLocalPileupUp2018Defaults25ns])

# Run-Dependent MC: DIGI step; 17 is a dummy test; 2018 with 2000 lumis
steps['DIGIPRMXUP17_PU25_RD']=merge([{'--procModifiers':'premix_stage2,runDependent'}, digiPremixUp2017Defaults25ns, { '--customise_commands':"\"process.GlobalTag.toGet = cms.VPSet( cms.PSet(     record = cms.string('EcalLaserAPDPNRatiosMCRcd'),  tag = cms.string('EcalLaserAPDPNRatios_UL_2017_mc'),  connect = cms.string('frontier://FrontierProd/CMS_CONDITIONS') ) )\""} ])

digiPremixRD2018 = {
    '--pileup_input':'das:/RelValPREMIXUP18_PU25/%s/PREMIX'%baseDataSetRelease[24],
    '--procModifiers':'premix_stage2,runDependent'
}

steps['DIGIPRMXUP18_PU25_RD']=merge([{'--conditions':'auto:phase1_2018_realistic_rd', '--relval': str(kevents_rd*1000) + ',' + str(events_per_job_rd), '--runsScenarioForMCIntegerWeights': 'Run2018_Equal_Lumi_Integer_Weights'}, digiPremixRD2018, digiPremixUp2018Defaults25ns])

# configuration to simulate cross run number boundary in IB, given 5 events per lumi
steps['DIGIPRMXUP18_PU25_RD_IB']=merge([{'--conditions':'auto:phase1_2018_realistic_rd', '--customise_commands':"\"process.source.setRunNumberForEachLumi = cms.untracked.vuint32(315257,316083) \"", '--procModifiers':'premix_stage2,runDependent'}, digiPremixRD2018, digiPremixUp2018Defaults25ns])

steps['DIGIPRMXLOCALUP18_PU25_RDPix']=merge([{'--conditions':'auto:phase1_2018_realistic_rd', '--procModifiers':'premix_stage2,runDependentForPixel'}, digiPremixLocalPileupUp2018Defaults25ns])

premixProd25ns = {'-s'             : 'DIGI,DATAMIX,L1,DIGI2RAW,HLT:@relval2016',
                 '--eventcontent' : 'PREMIXRAW',
                 '--datatier'     : 'PREMIXRAW'}
premixProd50ns = merge([{'-s':'DIGI,DATAMIX,L1,DIGI2RAW,HLT:@relval50ns'},premixProd25ns])
premixProd25ns2017 = merge([{'-s':'DIGI,DATAMIX,L1,DIGI2RAW,HLT:@relval2017'},premixProd25ns])
premixProd25ns2018 = merge([{'-s':'DIGI,DATAMIX,L1,DIGI2RAW,HLT:@relval2018','--datatier':'GEN-SIM-RAW','--geometry':'DB:Extended'},premixProd25ns])

steps['DIGIPRMXUP15_PROD_PU25']=merge([premixProd25ns,digiPremixUp2015Defaults25ns])
steps['DIGIPRMXUP15_PROD_PU50']=merge([premixProd50ns,digiPremixUp2015Defaults50ns])
steps['DIGIPRMXUP17_PROD_PU25']=merge([premixProd25ns2017,digiPremixUp2017Defaults25ns])
steps['DIGIPRMXUP18_PROD_PU25']=merge([premixProd25ns2018,digiPremixUp2018Defaults25ns])

dataReco={'--conditions':'auto:run1_data',
          '-s':'RAW2DIGI,L1Reco,RECO,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@standardDQMFakeHLT+@miniAODDQM',
          '--datatier':'RECO,MINIAOD,DQMIO',
          '--eventcontent':'RECO,MINIAOD,DQM',
          '--data':'',
          '--process':'reRECO',
          '--scenario':'pp',
          }

dataRecoAlCaCalo=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalCalZElectron+EcalCalWElectron+EcalUncalZElectron+EcalUncalWElectron+EcalTrg+HcalCalIsoTrk,DQM'}, dataReco])


hltKey='fake'
menu = autoHLT[hltKey]
steps['HLTD']=merge([{'--process':'reHLT',
                      '-s':'L1REPACK,HLT:@%s'%hltKey,
                      '--conditions':'auto:run1_hlt_%s'%menu,
                      '--data':'',
                      '--eventcontent': 'FEVTDEBUGHLT',
                      '--datatier': 'FEVTDEBUGHLT',
#                      '--output':'\'[{"e":"RAW","t":"RAW","o":["drop FEDRawDataCollection_rawDataCollector__LHC"]}]\'',
                      },])


steps['HLTDSKIM']=merge([{'--inputCommands':'"keep *","drop *_*_*_RECO"'},steps['HLTD']])
steps['HLTDSKIM2']=merge([{'--inputCommands':'"keep *","drop *_*_*_RECO","drop *_*_*_reRECO"'},steps['HLTD']])


steps['RECOD']=merge([{'--scenario':'pp',},dataReco])
steps['RECODR1']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias,DQM:@standardDQMFakeHLT+@miniAODDQM'},dataReco])
steps['RECODAlCaCalo']=merge([{'--scenario':'pp',},dataRecoAlCaCalo])

hltKey50ns='relval50ns'
steps['HLTDR2_50ns']=merge( [ {'-s':'L1REPACK,HLT:@%s'%hltKey50ns,},{'--conditions':'auto:run2_hlt_relval'},{'--era' : 'Run2_50ns'},steps['HLTD'] ] )

hltKey25ns='relval25ns'
steps['HLTDR2_25ns']=merge( [ {'-s':'L1REPACK:GT2,HLT:@%s'%hltKey25ns,},{'--conditions':'auto:run2_hlt_relval'},{'--era' : 'Run2_25ns'},steps['HLTD'] ] )

hltKey2016='relval2016'
steps['HLTDR2_2016']=merge( [ {'-s':'L1REPACK:Full,HLT:@%s'%hltKey2016,},{'--conditions':'auto:run2_hlt_relval'},{'--era' : 'Run2_2016'},steps['HLTD'] ] )

hltKey2017='relval2017'
steps['HLTDR2_2017']=merge( [ {'-s':'L1REPACK:Full,HLT:@%s'%hltKey2017,},{'--conditions':'auto:run2_hlt_relval'},{'--era' : 'Run2_2017'},steps['HLTD'] ] )

hltKey2018='relval2018'
steps['HLTDR2_2018']=merge( [ {'-s':'L1REPACK:Full,HLT:@%s'%hltKey2018,},{'--conditions':'auto:run2_hlt_relval'},{'--era' : 'Run2_2018'},steps['HLTD'] ] )

hltKey2022='relval2022'
steps['HLTDR3_2022']=merge( [ {'-s':'L1REPACK:Full,HLT:@%s'%hltKey2022,},{'--conditions':'auto:run3_hlt_relval'},{'--era' : 'Run3'},steps['HLTD'] ] )

hltKey2023='relval2023'
steps['HLTDR3_2023']=merge( [ {'-s':'L1REPACK:Full,HLT:@%s'%hltKey2023,},{'--conditions':'auto:run3_hlt_relval'},{'--era' : 'Run3_2023'},steps['HLTD'] ] )

steps['HLTDR3_2023B']=merge( [ {'-s':'L1REPACK:Full,HLT:@%s'%hltKey2023,},{'--conditions':'auto:run3_hlt_relval'},{'--era' : 'Run3'},steps['HLTD'] ] )

# special setting for lumi section boundary crossing in RunEGamma2018Dml
steps['HLTDR2_2018ml']=merge( [ {'--customise_commands':'"process.source.skipEvents=cms.untracked.uint32(7000)"'},steps['HLTDR2_2018'] ] )

steps['HLTDR2_2018_hBStar']=merge( [ {'--era' : 'Run2_2018_highBetaStar'},steps['HLTDR2_2018'] ] )

# use --era
steps['RECODR2_50ns']=merge([{'--scenario':'pp','--conditions':'auto:run2_data_relval','--era':'Run2_50ns',},dataReco])
steps['RECODR2_25ns']=merge([{'--scenario':'pp','--conditions':'auto:run2_data_relval','--era':'Run2_25ns','--customise':'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_25ns'},dataReco])
steps['RECODR2_2016']=merge([{'--scenario':'pp','--conditions':'auto:run2_data_relval','--era':'Run2_2016','--customise':'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2016'},dataReco])
steps['RECODR2_2017']=merge([{'--scenario':'pp','--conditions':'auto:run2_data_relval','--era':'Run2_2017','--customise':'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2017'},dataReco])
steps['RECODR2_2018']=merge([{'--scenario':'pp','--conditions':'auto:run2_data_relval','--era':'Run2_2018','--customise':'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018'},dataReco])

steps['RECODR2AlCaEle']=merge([{'--scenario':'pp','--conditions':'auto:run2_data_relval','--customise':'Configuration/DataProcessing/RecoTLR.customisePromptRun2',},dataRecoAlCaCalo])

steps['RECODSplit']=steps['RECOD'] # finer job splitting
steps['RECOSKIMALCA']=merge([{'--inputCommands':'"keep *","drop *_*_*_RECO"'
                              },steps['RECOD']])
steps['RECOSKIMALCAR1']=merge([{'--inputCommands':'"keep *","drop *_*_*_RECO"'
                                },steps['RECODR1']])
steps['REPACKHID']=merge([{'--scenario':'HeavyIons',
                         '-s':'RAW2DIGI,REPACK',
                         '--datatier':'RAW',
                         '--eventcontent':'REPACKRAW'},
                        steps['RECOD']])
steps['RECOHID10']=merge([{'--scenario':'HeavyIons',
                         '-s':'RAW2DIGI,L1Reco,RECO,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBiasHI+TkAlUpsilonMuMuHI+TkAlZMuMuHI+TkAlMuonIsolatedHI+TkAlJpsiMuMuHI+HcalCalMinBias,DQM:@standardDQMHIFakeHLT',
                         '--datatier':'RECO,DQMIO',
                         '--eventcontent':'RECO,DQM','-n':30},
                        steps['RECOD']])
steps['RECOHID11']=merge([{'--repacked':''},
                        steps['RECOHID10']])
steps['RECOHID10']['-s']+=',REPACK'
steps['RECOHID10']['--datatier']+=',RAW'
steps['RECOHID10']['--eventcontent']+=',REPACKRAW'

steps['HYBRIDRepackHI2015VR']={'--eventcontent':'RAW',
                               '--datatier':'RAW',
                               '--conditions':'auto:run2_hlt_hi',
                               '--step':'RAW2DIGI,REPACK:DigiToHybridRawRepack',
                               '--scenario':'HeavyIons',
                               '--data':'',
                               '--era':'Run2_HI',
                               '--customise':'RecoLocalTracker/SiStripZeroSuppression/customiseHybrid.addHybridEmulationBeforeRepack',
                               '--processName':'EMULATEHYBRID',
                               '--outputCommands':'"drop FEDRawDataCollection_*__HLT","drop FEDRawDataCollection_*__LHC"',
                               #'--customise_commands':'\'process.RAWoutput.outputCommands.append(\"drop *_*_*_HLT*\")\'',
                               '-n':100
                               }

steps['HYBRIDZSHI2015']=merge([{'--step': 'RAW2DIGI,REPACK:DigiToRawRepack',
                                '--processName':'REHLT',
                                '--customise': 'RecoLocalTracker/SiStripZeroSuppression/customiseHybrid.runOnHybridZS,RecoLocalTracker/SiStripZeroSuppression/customiseHybrid.repackZSHybrid',
                                '--customise_commands':'\'from Configuration.Applications.ConfigBuilder import MassReplaceInputTag; MassReplaceInputTag(process, new="hybridRawDataRepacker")\'' ,
                                },steps['HYBRIDRepackHI2015VR']])

steps['RECOHID15']=merge([{ '--conditions':'auto:run2_data',
                            '--era':'Run2_HI'
                            },steps['RECOHID11']])

steps['RECOHID18']=merge([{ '--scenario':'pp',
                            '--conditions':'auto:run2_data_promptlike_hi',
                            '-s':'RAW2DIGI,L1Reco,RECO,ALCA:SiStripCalZeroBias+SiPixelCalZeroBias,SKIM:PbPbEMu+PbPbZEE+PbPbZMM+PbPbZMu,DQM:@commonFakeHLT+@standardDQMFakeHLT',
                            '--datatier':'AOD,DQMIO',
                            '--eventcontent':'AOD,DQM',
                            '--era':'Run2_2018_pp_on_AA',
                            '--customise':'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018_pp_on_AA',
                            '-n':'10'
                            },steps['RECOHID15']])

steps['RAWPRIMEHI18']={ '--scenario':'pp',
                        '--conditions':'auto:run2_data_promptlike_hi',
                        '-s':'REPACK:DigiToApproxClusterRaw',
                        '--datatier':'GEN-SIM-DIGI-RAW-HLTDEBUG',
                        '--eventcontent':'REPACKRAW',
                        '--era':'Run2_2018_pp_on_AA',
                        '--customise':'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018_pp_on_AA',
                        '-n':'10',
                        '--procModifiers':'approxSiStripClusters',
                        '--customise_commands':'\"process.rawPrimeDataRepacker.src=\'rawDataRepacker\'\"',
                        '--repacked':'',
                        '--process':'REHLT'
}

steps['RAWPRIMEHI22']={ '--scenario':'pp',
                        '--conditions':'auto:run3_data_prompt',
                        '-s':'REPACK:DigiToApproxClusterRaw',
                        '--datatier':'GEN-SIM-DIGI-RAW-HLTDEBUG',
                        '--era':'Run3_pp_on_PbPb_approxSiStripClusters',
                        '--eventcontent':'REPACKRAW',
                        '-n':'10',
                        '--customise_commands':'\"process.rawPrimeDataRepacker.src=\'rawDataRepacker\'\"',
                        '--repacked':'',
                        '--process':'REHLT'
}

steps['RAWPRIMESIMHI18']={ '--scenario':'pp',
                           '--conditions':'auto:phase1_2022_realistic_hi',
                           '-s':'REPACK:DigiToApproxClusterRaw',
                           '--datatier':'GEN-SIM-DIGI-RAW-HLTDEBUG',
                           '--eventcontent':'FEVTDEBUGHLT',
                           '--era':'Run2_2018_pp_on_AA',
                           '-n':'10',
                           '--procModifiers':'approxSiStripClusters',
                           '--customise_commands':'\"process.hltSiStripRawToDigi.ProductLabel=\'rawDataCollector\';process.hltScalersRawToDigi.scalersInputTag=\'rawDataCollector\'\"',
                           '--process':'REHLT'
}

steps['RECOHID18APPROXCLUSTERS']=merge([{ '--scenario':'pp',
                                          '--conditions':'auto:run2_data_promptlike_hi',
                                          '-s':'RAW2DIGI,L1Reco,RECO,ALCA:SiStripCalZeroBias+SiPixelCalZeroBias,SKIM:PbPbEMu+PbPbZEE+PbPbZMM+PbPbZMu,DQM:@commonFakeHLT+@standardDQMFakeHLT',
                                          '--datatier':'AOD,DQMIO',
                                          '--eventcontent':'AOD,DQM',
                                          '--era':'Run2_2018_pp_on_AA',
                                          '--customise':'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2018_pp_on_AA',
                                          '--procModifiers':'approxSiStripClusters',
                                          '-n':'10'
                                      },steps['RECOHID15']])

steps['REMINIAODHID18']={ '--scenario':'pp',
                          '--conditions':'auto:run2_data_promptlike_hi',
                          '-s':'PAT,DQM:@miniAODDQM',
                          '--datatier':'MINIAOD,DQMIO',
                          '--eventcontent':'MINIAOD,DQM',
                          '--era':'Run2_2018_pp_on_AA',
                          '--procModifiers':'run2_miniAOD_pp_on_AA_103X',
                          '--data':'',
                          '--processName':'PAT',
                          '-n':'100'
}


steps['RECOHID22APPROXCLUSTERS']=merge([{ '--scenario':'pp',
                                          '--conditions':'auto:run3_data_prompt',
                                          '-s':'RAW2DIGI,L1Reco,RECO,DQM:@commonFakeHLT+@standardDQMFakeHLT',
                                          '--datatier':'AOD,DQMIO',
                                          '--eventcontent':'AOD,DQM',
                                          '--era':'Run3_pp_on_PbPb_approxSiStripClusters',
                                          '--repacked':'',
                                          '-n':'100'
                                      },steps['RECOHID15']])

steps['RECOHID22']=merge([{ '--scenario':'pp',
                            '--conditions':'auto:run3_data_prompt',
                            '-s':'RAW2DIGI,L1Reco,RECO,DQM:@commonFakeHLT+@standardDQMFakeHLT',
                            '--datatier':'AOD,DQMIO',
                            '--eventcontent':'AOD,DQM',
                            '--era':'Run3_pp_on_PbPb',
                            '--repacked':'',
                            '-n':'100'
                        },steps['RECOHID15']])


steps['TIER0']=merge([{'--customise':'Configuration/DataProcessing/RecoTLR.customisePrompt',
                       '-s':'RAW2DIGI,L1Reco,RECO,ALCAPRODUCER:@allForPrompt,DQM:@allForPrompt,ENDJOB',
                       '--datatier':'RECO,AOD,ALCARECO,DQMIO',
                       '--eventcontent':'RECO,AOD,ALCARECO,DQM',
                       '--process':'RECO'
                       },dataReco])
steps['TIER0EXP']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCAPRODUCER:@allForExpress+AlCaPCCZeroBiasFromRECO+AlCaPCCRandomFromRECO,DQM:@express,ENDJOB',
                          '--datatier':'ALCARECO,DQMIO',
                          '--eventcontent':'ALCARECO,DQM',
                          '--customise':'Configuration/DataProcessing/RecoTLR.customiseExpress',
                          },steps['TIER0']])

steps['TIER0EXPRUN2']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCAPRODUCER:@allForExpress+AlCaPCCZeroBiasFromRECO+AlCaPCCRandomFromRECO,DQM:@express,ENDJOB',
                          '--process':'RECO',
                          '--datatier':'ALCARECO,DQMIO',
                          '--eventcontent':'ALCARECO,DQM',
                          '--customise':'Configuration/DataProcessing/RecoTLR.customiseExpress',
                          '--era':'Run2_2017',
                          '--conditions':'auto:run2_data'
                          },steps['TIER0']])
steps['TIER0EXPRUN3']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCAPRODUCER:@allForExpress,DQM:@express,ENDJOB',
                          '--process':'RECO',
                          '--datatier':'ALCARECO,DQMIO',
                          '--eventcontent':'ALCARECO,DQM',
                          '--customise':'Configuration/DataProcessing/RecoTLR.customiseExpress',
                          '--era':'Run3',
                          '--conditions':'auto:run3_data_express'
                          },steps['TIER0']])
steps['TIER0EXPCOSMICSRUN3']=merge([{'--scenario':'cosmics',
                                     '-s':'RAW2DIGI,L1Reco,RECO,ALCAPRODUCER:@allForExpressCosmics,ENDJOB',
                                     '-n':1000,
                                     '--data':'',
                                     '--process':'RECO',
                                     '--era':'Run3',
                                     '--conditions':'auto:run3_data_express',
                                     '--datatier':'ALCARECO',
                                     '--eventcontent':'ALCARECO',
                                     '--customise_commands':'"process.source.eventsToProcess = cms.untracked.VEventRange(\'360336:24:422152-360336:24:422152\')"'},steps['TIER0EXP']])

steps['TIER0PROMPTRUN3']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCAPRODUCER:@allForPrompt+@allForExpress,DQM:@allForPrompt,ENDJOB',
                          '--process':'RECO',
                          '--datatier':'ALCARECO,DQMIO',
                          '--eventcontent':'ALCARECO,DQM',
                          '--customise':'Configuration/DataProcessing/RecoTLR.customisePrompt',
                          '--era':'Run3',
                          '--conditions':'auto:run3_data_prompt'
                          },steps['TIER0']])

steps['TIER0EXPHI']={      '--conditions':'auto:run1_data',
          '-s':'RAW2DIGI,L1Reco,RECO,ALCAPRODUCER:@allForExpressHI,DQM,ENDJOB',
          '--datatier':'ALCARECO,DQMIO',
          '--eventcontent':'ALCARECO,DQM',
          '--data':'',
          '--process':'RECO',
          '--scenario':'HeavyIons',
          '--customise':'Configuration/DataProcessing/RecoTLR.customiseExpressHI',
          '--repacked':'',
          '-n':'10'
                          }
steps['TIER0EXPTE']={'-s': 'ALCAPRODUCER:EcalTestPulsesRaw',
                     '--conditions': 'auto:run2_data',
                     '--datatier':'ALCARECO',
                     '--eventcontent':'ALCARECO',
                     '--data': '',
                     '--process': 'RECO',
                     '--scenario': 'pp',
                     #'--customise':'Configuration/DataProcessing/RecoTLR.customiseExpress',
                     }

steps['TIER0EXPLP']={'-s': 'ALCAPRODUCER:AlCaPCCRandom',
                        '--conditions': 'auto:run3_data_express',
                        '--era':'Run3',
                        '--datatier':'ALCARECO',
                        '--eventcontent':'ALCARECO',
                        '--data': '',
                        '--scenario': 'pp',
                        '--process': 'RECO',
                        # '--customise':'Configuration/DataProcessing/RecoTLR.customiseExpress',
                        }

steps['TIER0PROMPTLP']={'-s': 'ALCAPRODUCER:AlCaPCCZeroBias+RawPCCProducer',
                        '--conditions': 'auto:run3_data_prompt',
                        '--era':'Run3',
                        '--datatier':'ALCARECO',
                        '--eventcontent':'ALCARECO',
                        '--data': '',
                        '--scenario': 'pp',
                        '--process': 'RECO',
                        '--filein': 'filelist:step1_dasquery.log'
                        # '--customise':'Configuration/DataProcessing/RecoTLR.customiseExpress',
                        }


steps['ALCAEXPLP']={'-s':'ALCAOUTPUT:AlCaPCCRandom,ALCA:PromptCalibProdLumiPCC',
                  '--conditions': 'auto:run3_data_express',
                  '--era':'Run3',
                  '--datatier':'ALCARECO',
                  '--eventcontent':'ALCARECO',
                  '--triggerResultsProcess': 'RECO'}

steps['ALCAHARVLP']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProdLumiPCC']),
                     '--conditions': 'auto:run3_data_express',
                     '--era':'Run3',
                     '--scenario':'pp',
                     '--datatier':'DQM',
                     '--eventcontent': 'DQM',
                     '--data':'',
                     '--filein':'file:PromptCalibProdLumiPCC.root'}


steps['TIER0EXPHPBS']={'-s':'RAW2DIGI,L1Reco,RECO:reconstruction_trackingOnly,ALCAPRODUCER:TkAlMinBias,DQM:DQMOfflineTracking,ENDJOB',
                          '--process':'RECO',
                          '--scenario': 'pp',
                          '--era':'Run2_2017',
                          '--conditions':'auto:run2_data',
                          '--data': '',
                          '--datatier':'ALCARECO,DQMIO',
                          '--eventcontent':'ALCARECO,DQM',
                          '--customise':'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2017_express_trackingOnly',
                          }

steps['TIER0RAWSIPIXELCAL']={'-s':'RAW2DIGI,L1Reco,RECO,ALCAPRODUCER:SiPixelCalZeroBias,DQM:@express,ENDJOB',
                          '--process':'RECO',
                          '--scenario': 'pp',
                          '--era':'Run2_2017',
                          '--conditions':'auto:run2_data',
                          '--data': '',
                          '--datatier':'ALCARECO,DQMIO',
                          '--eventcontent':'ALCARECO,DQM',
                          '--customise':'Configuration/DataProcessing/RecoTLR.customiseExpress',
                          }

steps['TIER0EXPSIPIXELCAL']={'-s':'RAW2DIGI,L1Reco,ALCAPRODUCER:SiPixelCalZeroBias,ENDJOB',
                          '--process':'ALCARECO',
                          '--scenario': 'pp',
                          '--era':'Run2_2017',
                          '--conditions':'auto:run2_data',
                          '--data': '',
                          '--datatier':'ALCARECO',
                          '--eventcontent':'ALCARECO',
                          }

steps['TIER0EXPPPSCAL']={'-s':'RAW2DIGI,L1Reco,ALCAPRODUCER:PPSCalTrackBasedSel,ENDJOB',
                          '--process':'ALCARECO',
                          '--scenario': 'pp',
                          '--era':'Run2_2017',
                          '--conditions':'auto:run2_data',
                          '--data': '',
                          '--datatier':'ALCARECO',
                          '--eventcontent':'ALCARECO',
                          }
                          
steps['TIER0EXPPPSCALRUN3']={'-s':'RAW2DIGI,L1Reco,ALCAPRODUCER:PPSCalMaxTracks,ENDJOB',
                          '-n':1000,
                          '--process':'ALCARECO',
                          '--scenario': 'pp',
                          '--era':'Run3',
                          '--conditions':'auto:run3_data_express',
                          '--data': '',
                          '--datatier':'ALCARECO',
                          '--eventcontent':'ALCARECO',
                          }

steps['TIER0EXPPPSCALALIG']={'-s':'RAW2DIGI,L1Reco,ALCAPRODUCER:PPSCalTrackBasedSel,ENDJOB',
                             '-n':1000,
                             '--process':'ALCARECO',
                             '--scenario': 'pp',
                             '--era':'Run2_2017',
                             '--conditions':'auto:run2_data',
                             '--data': '',
                             '--datatier':'ALCARECO',
                             '--eventcontent':'ALCARECO',
                             }

steps['ALCASPLITHPBS']={'-s':'ALCAOUTPUT:TkAlMinBias,ALCA:PromptCalibProdBeamSpotHP+PromptCalibProdBeamSpotHPLowPU',
                        '--scenario':'pp',
                        '--data':'',
                        '--era':'Run2_2017',
                        '--datatier':'ALCARECO',
                        '--eventcontent':'ALCARECO',
                        '--conditions':'auto:run2_data',
                        '--triggerResultsProcess':'RECO',
                        }

steps['ALCASPLITSIPIXELCAL']={'-s':'ALCAOUTPUT:SiPixelCalZeroBias,ALCA:PromptCalibProdSiPixel',
                        '--scenario':'pp',
                        '--data':'',
                        '--era':'Run2_2017',
                        '--datatier':'ALCARECO',
                        '--eventcontent':'ALCARECO',
                        '--conditions':'auto:run2_data',
                        '--triggerResultsProcess':'RECO',
                        #'--filein':'file:step2.root'
                        }

steps['ALCASPLITPPSCAL']={'-s':'ALCAOUTPUT:PPSCalTrackBasedSel,ALCA:PromptCalibProdPPSTimingCalib',
                        '--scenario':'pp',
                        '--data':'',
                        '--era':'Run2_2017',
                        '--datatier':'ALCARECO',
                        '--eventcontent':'ALCARECO',
                        '--conditions':'auto:run2_data',
                        '--triggerResultsProcess':'RECO',
                        }
                        
steps['ALCASPLITPPSCALRUN3']={'-s':'ALCAOUTPUT:PPSCalMaxTracks,ALCA:PromptCalibProdPPSTimingCalib',
			 '-n':1000,
                        '--scenario':'pp',
                        '--data':'',
                        '--era':'Run3',
                        '--datatier':'ALCARECO',
                        '--eventcontent':'ALCARECO',
                        '--conditions':'auto:run3_data_express',
                        '--triggerResultsProcess':'ALCARECO',
                        }                     
                        
steps['ALCASPLITPPSCALSAMPIC']={'-s':'ALCAOUTPUT:PPSCalMaxTracks,ALCA:PromptCalibProdPPSDiamondSampic',
                        '--scenario':'pp',
                        '--data':'',
                        '--era':'Run3',
                        '--datatier':'ALCARECO',
                        '--eventcontent':'ALCARECO',
                        '--conditions':'auto:run3_data_express',
                        '--triggerResultsProcess':'ALCARECO',
                        }
                        

steps['ALCASPLITPPSALIG']={'-s':'ALCAOUTPUT:PPSCalTrackBasedSel,ALCA:PromptCalibProdPPSAlignment',
                           '-n':1000,
                           '--scenario':'pp',
                           '--data':'',
                           '--era':'Run2_2017',
                           '--datatier':'ALCARECO',
                           '--eventcontent':'ALCARECO',
                           '--conditions':'auto:run3_data_express',
                           '--triggerResultsProcess':'RECO',
                           }
                           
steps['ALCASPLITPPSALIGRUN3']={'-s':'ALCAOUTPUT:PPSCalMaxTracks,ALCA:PromptCalibProdPPSAlignment',
                           '-n':1000,
                           '--scenario':'pp',
                           '--data':'',
                           '--era':'Run3',
                           '--datatier':'ALCARECO',
                           '--eventcontent':'ALCARECO',
                           '--conditions':'auto:run3_data_express',
                           '--triggerResultsProcess':'ALCARECO',
                           }

steps['ALCASPLITSIPIXELLAMCS']={'-s':'ALCA:PromptCalibProdSiPixelLAMCS',
                                '-n':1000,
                                '--scenario':'cosmics',
                                '--era':'Run3',
                                '--datatier':'ALCARECO',
                                '--eventcontent':'ALCARECO',
                                '--conditions':'auto:run3_data_express',
                                '--customise_commands':'"process.ALCARECOCalCosmicsFilterForSiPixelLorentzAngleMCS.HLTPaths = [\'*\']; process.ALCARECOCalCosmicsFilterForSiPixelLorentzAngleMCS.TriggerResultsTag = cms.InputTag(\'TriggerResults\',\'\',\'HLT\')"'}

steps['ALCAHARVDHPBS']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProdBeamSpotHP']),
                        #'--conditions':'auto:run2_data_promptlike',
                        '--conditions':'auto:run3_data_express', # to replaced with line above once run2_data_promptlike will contain DropBoxMetadata
                        '--scenario':'pp',
                        '--data':'',
                        '--era':'Run2_2017',
                        '--customise':'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2017_harvesting_trackingOnly',
                        '--filein':'file:PromptCalibProdBeamSpotHP.root'}

steps['ALCAHARVDHPBSLOWPU']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProdBeamSpotHPLowPU']),
                        #'--conditions':'auto:run2_data_promptlike',
                        '--conditions':'auto:run3_data_express', # to replaced with line above once run2_data_promptlike will contain DropBoxMetadata
                        '--scenario':'pp',
                        '--data':'',
                        '--era':'Run2_2017',
                        '--customise':'Configuration/DataProcessing/RecoTLR.customisePostEra_Run2_2017_harvesting_trackingOnly',
                        '--filein':'file:PromptCalibProdBeamSpotHPLowPU.root'}


steps['ALCAHARVDSIPIXELCAL']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProdSiPixel']),
                        '--conditions':'auto:run3_data_express',
                        '--scenario':'pp',
                        '--data':'',
                        '--era':'Run2_2017',
                        '--filein':'file:PromptCalibProdSiPixel.root'}

steps['ALCAHARVDSIPIXELCALRUN1']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProdSiPixel']),
                        '--conditions':'auto:run1_data',
                        '--scenario':'pp',
                        '--data':'',
                        '--filein':'file:PromptCalibProdSiPixel.root'}

steps['ALCAHARVDSIPIXELCALLA']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProdSiPixelLA']),
                        '--conditions':'auto:run3_data_express',
                        '--scenario':'pp',
                        '--data':'',
                        '--era':'Run2_2017',
                        '--filein':'file:PromptCalibProdSiPixelLA.root'}
                        
steps['ALCAHARVDSIPIXELALIHG']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProdSiPixelAliHG']),
                     '--conditions':'auto:run2_data',
                     '--scenario':'pp',
                     '--data':'',
                     '--filein':'file:PromptCalibProdSiPixelAliHG.root',
                     '--customise':'Alignment/CommonAlignmentProducer/customizeLSNumberFilterForRelVals.lowerHitsPerStructure'}

steps['ALCAHARVDPPSCAL']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProdPPSTimingCalib']),
                        '--conditions':'auto:run3_data_express',
                        '--scenario':'pp',
                        '--data':'',
                        '--era':'Run2_2017',
                        '--filein':'file:ALCARECOStreamPromptCalibProdPPSTimingCalib.root'}
                        
steps['ALCAHARVDPPSCALRUN3']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProdPPSTimingCalib']),
                        '--conditions':'auto:run3_data_express',
                        '--scenario':'pp',
                        '--data':'',
                        '--era':'Run3',
                        '--filein':'file:ALCARECOStreamPromptCalibProdPPSTimingCalib.root'}

steps['ALCAHARVDPPSCALSAMPIC']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProdPPSDiamondSampic']),
                        '--conditions':'auto:run3_data_express',
                        '--scenario':'pp',
                        '--data':'',
                        '--era':'Run3',
                        '--filein':'file:ALCARECOStreamPromptCalibProdPPSDiamondSampic.root'}
                        
steps['ALCAHARVDPPSALIG']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProdPPSAlignment']),
                           '--conditions':'auto:run3_data_express',
                           '--scenario':'pp',
                           '--data':'',
                           '--era':'Run2_2017',
                           '--filein':'file:PromptCalibProdPPSAlignment.root'}
                           
steps['ALCAHARVDPPSALIGRUN3']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProdPPSAlignment']),
                           '--conditions':'auto:run3_data_express',
                           '--scenario':'pp',
                           '--data':'',
                           '--era':'Run3',
                           '--filein':'file:PromptCalibProdPPSAlignment.root'}

steps['ALCAHARVDSIPIXELLAMCS']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProdSiPixelLAMCS']),
                           '--conditions':'auto:run3_data_express',
                           '--scenario':'cosmics',
                           '--data':'',
                           '--era':'Run3',
                           '--filein':'file:PromptCalibProdSiPixelLAMCS.root'}

steps['RECOCOSD']=merge([{'--scenario':'cosmics',
                          '-s':'RAW2DIGI,L1Reco,RECO,DQM',
                          '--datatier':'RECO,DQMIO',     # no miniAOD for cosmics
                          '--eventcontent':'RECO,DQM',
                          '--customise':'Configuration/DataProcessing/RecoTLR.customiseCosmicData'
                          },dataReco])

steps['RECOCOSDRUN2']=merge([{'--conditions':'auto:run2_data','--era':'Run2_2016'},steps['RECOCOSD']])

# Run 3
steps['RECODR3']=merge([{'--scenario':'pp',
                         '-s':'RAW2DIGI,L1Reco,RECO,DQM',
                         '--conditions':'auto:run3_data',
                         '--era':'Run3',
                         '--customise':'Configuration/DataProcessing/RecoTLR.customisePostEra_Run3'},dataReco])

steps['RECODR3_2023']=merge([{'--era':'Run3_2023'},steps['RECODR3']])
                         
steps['RECODR3_reHLT_2022']=merge([{'--conditions':'auto:run3_data_relval', '--hltProcess':'reHLT'},steps['RECODR3']])
steps['RECODR3_reHLT_2023']=merge([{'--conditions':'auto:run3_data_prompt_relval', '--hltProcess':'reHLT'},steps['RECODR3_2023']])
steps['RECODR3_reHLT_2023B']=merge([{'--conditions':'auto:run3_data_prompt_relval', '--hltProcess':'reHLT'},steps['RECODR3']])


steps['RECODR3Splash']=merge([{'-n': 2,
                               '-s': 'RAW2DIGI,L1Reco,RECO,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@standardDQMFakeHLT+@miniAODDQM'
                              },steps['RECODR3']])

steps['RECOCOSDRUN3']=merge([{'--conditions':'auto:run3_data','--era':'Run3'},steps['RECOCOSD']])

steps['RECOCOSDPROMPTRUN3']=merge([{'--conditions':'auto:run3_data_prompt',
                                    '--era':'Run3',
                                    '--customise':'Configuration/DataProcessing/RecoTLR.customisePrompt,Configuration/DataProcessing/RecoTLR.customiseCosmicData'
                                },steps['RECOCOSD']])

steps['RECOCOSDEXPRUN3']=merge([{'--conditions':'auto:run3_data_express',
                                 '--era':'Run3',
                                 '--customise':'Configuration/DataProcessing/RecoTLR.customiseExpress,Configuration/DataProcessing/RecoTLR.customiseCosmicData'
                             },steps['RECOCOSD']])

steps['RECOALCAECALPHISYMDR3']=merge([{'--scenario':'pp',
                                       '--era':'Run3',
                                       '--conditions':'auto:run3_data_prompt',
                                       '--datatier':'RECO',
                                       '--eventcontent':'RECO',
                                       '-n':'-1',
                                       '-s':'RECO:bunchSpacingProducer+ecalMultiFitUncalibRecHitTask+ecalCalibratedRecHitTask',
                                       '--customise':'Calibration/EcalCalibAlgos/EcalPhiSymRecoSequence_cff'},dataReco])

# step1 gensim for HI mixing
step1Up2018HiMixDefaults = merge ([{'--beamspot':'MatchHI', '--pileup':'HiMixGEN', '--scenario':'HeavyIons'},hiDefaults2018_ppReco,PUHI,step1Up2018HiProdDefaults])
def gen2018HiMix(fragment,howMuch):
    global step1Up2018HiMixDefaults
    return merge([{'cfg':fragment},howMuch,step1Up2018HiMixDefaults])

step1Up2022HiMixDefaults = merge ([{'--beamspot':'MatchHI', '--pileup':'HiMixGEN', '--scenario':'HeavyIons'},hiDefaults2022_ppReco,PUHI2022,step1Up2022HiProdDefaults])
def gen2022HiMix(fragment,howMuch):
    global step1Up2022HiMixDefaults
    return merge([{'cfg':fragment},howMuch,step1Up2022HiMixDefaults])

step1Up2023HiMixDefaults = merge ([{'--beamspot':'MatchHI', '--pileup':'HiMixGEN', '--scenario':'HeavyIons'},hiDefaults2023_ppReco,PUHI2023,step1Up2023HiProdDefaults])
def gen2023HiMix(fragment,howMuch):
    global step1Up2023HiMixDefaults
    return merge([{'cfg':fragment},howMuch,step1Up2023HiMixDefaults])

steps['Pyquen_GammaJet_pt20_2760GeV']=gen2018HiMix('Pyquen_GammaJet_pt20_2760GeV_cfi',Kby(9,100))
steps['Pyquen_DiJet_pt80to120_2760GeV']=gen2018HiMix('Pyquen_DiJet_pt80to120_2760GeV_cfi',Kby(9,100))
steps['Pyquen_ZeemumuJets_pt10_2760GeV']=gen2018HiMix('Pyquen_ZeemumuJets_pt10_2760GeV_cfi',Kby(9,100))
steps['Pyquen_GammaJet_pt20_2760GeV_2022']=gen2022HiMix('Pyquen_GammaJet_pt20_2760GeV_cfi',Kby(9,100))
steps['Pyquen_DiJet_pt80to120_2760GeV_2022']=gen2022HiMix('Pyquen_DiJet_pt80to120_2760GeV_cfi',Kby(9,100))
steps['Pyquen_ZeemumuJets_pt10_2760GeV_2022']=gen2022HiMix('Pyquen_ZeemumuJets_pt10_2760GeV_cfi',Kby(9,100))
steps['Pyquen_GammaJet_pt20_5362GeV_2023']=gen2023HiMix('Pyquen_GammaJet_pt20_5362GeV_cfi',Kby(9,100))
steps['Pyquen_DiJet_pt80to120_5362GeV_2023']=gen2023HiMix('Pyquen_DiJet_pt80to120_5362GeV_cfi',Kby(9,100))
steps['Pyquen_ZeemumuJets_pt10_5362GeV_2023']=gen2023HiMix('Pyquen_ZeemumuJets_pt10_5362GeV_cfi',Kby(9,100))

# step3
step3Defaults = {
                  '-s'            : 'RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM',
                  '--conditions'  : 'auto:run1_mc',
                  '--no_exec'     : '',
                  '--datatier'    : 'GEN-SIM-RECO,MINIAODSIM,DQMIO',
                  '--eventcontent': 'RECOSIM,MINIAODSIM,DQM',
                  }
step3DefaultsAlCaCalo=merge([{'-s':'RAW2DIGI,L1Reco,RECO,PAT,ALCA:EcalCalZElectron+EcalCalWElectron+EcalUncalZElectron+EcalUncalWElectron+HcalCalIsoTrk,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM'},step3Defaults])

steps['DIGIPU']=merge([{'--process':'REDIGI'},steps['DIGIPU1']])

#for 2015
step3Up2015Defaults = {
    #'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
    '-s':'RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM',
    '--conditions':'auto:run2_mc',
    '-n':'10',
    '--datatier':'GEN-SIM-RECO,MINIAODSIM,DQMIO',
    '--eventcontent':'RECOSIM,MINIAODSIM,DQM',
    '--era' : 'Run2_2016'
    }

step3Up2015Defaults50ns = merge([{'-s':'RAW2DIGI,L1Reco,RECO,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM','--conditions':'auto:run2_mc_50ns','--era':'Run2_50ns'},step3Up2015Defaults])

step3Up2015DefaultsAlCaCalo = merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCA:EcalCalZElectron+EcalCalWElectron+EcalUncalZElectron+EcalUncalWElectron+EcalTrg+HcalCalIsoTrk,VALIDATION:@standardValidationNoHLT,DQM:@standardDQMFakeHLT'},step3Up2015Defaults])
step3Up2015DefaultsAlCaCalo50ns = merge([{'--conditions':'auto:run2_mc_50ns','--era':'Run2_50ns'},step3Up2015DefaultsAlCaCalo])

step3Up2015Hal = {'-s'            :'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                  '--conditions'   :'auto:run2_mc',
                  '--datatier'     :'GEN-SIM-RECO,DQMIO',
                  '--eventcontent':'RECOSIM,DQM',
                  '-n'            :'10',
                  '--era'          :'Run2_2016'
                  }

step3_pixel_ntuplet_cpu = {
    '--procModifiers': 'pixelNtupletFit'
}
step3_pixel_ntuplet_gpu = {
    '--procModifiers': 'pixelNtupletFit,gpu'
}
step3_pixel_ntuplet_gpu_validation = {
    '--procModifiers': 'pixelNtupletFit,gpuValidation'
}
step3_pixel_triplets = {
    '--customise': 'RecoTracker/Configuration/customizePixelTracksForTriplets.customizePixelTracksForTriplets'
}
step3_pixel_profiling = {
    '--customise': 'RecoTracker/Configuration/customizePixelOnlyForProfiling.customizePixelOnlyForProfilingGPUOnly'
}
step3_pixel_triplets_profiling = {
    '--customise': 'RecoTracker/Configuration/customizePixelTracksForTriplets.customizePixelTracksForTriplets,RecoTracker/Configuration/customizePixelOnlyForProfiling.customizePixelOnlyForProfilingGPUOnly'
}
step3_gpu = {
    '--procModifiers': 'gpu',
}
step3_trackingLowPU = {
    '--era': 'Run2_2016_trackingLowPU'
}
step3_HIPM = {
    '--era': 'Run2_2016_HIPM'
}
step3Up2015Defaults_trackingOnly = merge([step3_trackingOnly, step3Up2015Defaults])

steps['RECOUP15']=merge([step3Up2015Defaults]) # todo: remove UP from label
steps['RECOUP15_L1TEgDQM']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM+@L1TEgamma'},step3Up2015Defaults]) # todo: remove UP from label
steps['RECOUP15_L1TMuDQM']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM+@L1TMuon'},step3Up2015Defaults]) # todo: remove UP from label
steps['RECOUP15AlCaCalo']=merge([step3Up2015DefaultsAlCaCalo]) # todo: remove UP from label

steps['RECOUP15_trackingOnly']=merge([step3Up2015Defaults_trackingOnly]) # todo: remove UP from label
steps['RECOUP15_trackingLowPU']=merge([step3_trackingLowPU, step3Up2015Defaults]) # todo: remove UP from label
steps['RECOUP15_trackingOnlyLowPU']=merge([step3_trackingLowPU, step3Up2015Defaults_trackingOnly]) # todo: remove UP from label
steps['RECOUP15_HIPM']=merge([step3_HIPM,step3Up2015Defaults]) # todo: remove UP from label
steps['RECOUP15_BPHDQM']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM+@heavyFlavor'},step3Up2015Defaults]) # todo: remove UP from label

steps['RECOUP17']=merge([{'--conditions':'auto:phase1_2017_realistic','--era' : 'Run2_2017','--geometry' : 'DB:Extended'},steps['RECOUP15']])
steps['RECOUP17_PU25']=merge([PU25UP17,steps['RECOUP17']])

steps['RECOUP18']=merge([{'--conditions':'auto:phase1_2018_realistic','--era' : 'Run2_2018','--geometry' : 'DB:Extended'},steps['RECOUP15']])
steps['RECOUP18_PU25']=merge([PU25UP18,steps['RECOUP18']])

# for Run1 PPb data workflow
steps['RECO_PPbData']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCA:TkAlMinBias+TkAlMuonIsolatedPA+TkAlUpsilonMuMuPA+TkAlZMuMuPA,SKIM:PAZMM+PAZEE+PAMinBias,DQM','--scenario':'pp','--conditions':'auto:run1_data','--era':'Run1_pA','--datatier':'AOD,DQMIO','--eventcontent':'AOD,DQM'}, dataReco])

# for Run2 PPb MC workflow
steps['RECOUP15_PPb']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCA:TkAlMinBias+TkAlMuonIsolatedPA+TkAlUpsilonMuMuPA+TkAlZMuMuPA,VALIDATION,DQM','--conditions':'auto:run2_mc_pa','--era':'Run2_2016_pA','--datatier':'AODSIM,DQMIO','--eventcontent':'AODSIM,DQM'}, steps['RECOUP15']])

#steps['RECOUP15PROD1']=merge([{ '-s' : 'RAW2DIGI,L1Reco,RECO,DQM:DQMOfflinePOGMC', '--datatier' : 'AODSIM,DQMIO', '--eventcontent' : 'AODSIM,DQM'},step3Up2015Defaults])

steps['RECODreHLT']=merge([{'--hltProcess':'reHLT','--conditions':'auto:run1_data_%s'%menu},steps['RECOD']])
steps['RECODR1reHLT']=merge([{'--hltProcess':'reHLT','--conditions':'auto:run1_data_%s'%menu},steps['RECODR1']])
steps['RECODR1reHLT2']=merge([{'--process':'reRECO2'},steps['RECODR1reHLT']])

steps['RECODreHLTAlCaCalo']=merge([{'--hltProcess':'reHLT','--conditions':'auto:run1_data_%s'%menu},steps['RECODAlCaCalo']])

steps['RECODR2_25nsreHLT']=merge([{'--hltProcess':'reHLT'},steps['RECODR2_25ns']])
steps['RECODR2_25nsreHLT_ZB']=merge([{'--hltProcess':'reHLT','-s':'RAW2DIGI,L1Reco,RECO,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@rerecoZeroBiasFakeHLT+@miniAODDQM'},steps['RECODR2_25ns']])
steps['RECODR2_50nsreHLT']=merge([{'--hltProcess':'reHLT'},steps['RECODR2_50ns']])
steps['RECODR2_50nsreHLT_ZB']=merge([{'--hltProcess':'reHLT','-s':'RAW2DIGI,L1Reco,RECO,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@rerecoZeroBiasFakeHLT+@miniAODDQM'},steps['RECODR2_50ns']])

steps['RECODR2_2016reHLT']=merge([{'--hltProcess':'reHLT','--conditions':'auto:run2_data_relval'},steps['RECODR2_2016']])
steps['RECODR2_2016reHLT_ZB']=merge([{'--hltProcess':'reHLT','--conditions':'auto:run2_data_relval','-s':'RAW2DIGI,L1Reco,RECO,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@rerecoZeroBiasFakeHLT+@miniAODDQM'},steps['RECODR2_2016']])
steps['RECODR2_2016reHLT_L1TEgDQM']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@standardDQMFakeHLT+@miniAODDQM+@L1TEgamma'},steps['RECODR2_2016reHLT']])
steps['RECODR2_2016reHLT_L1TMuDQM']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@standardDQMFakeHLT+@miniAODDQM+@L1TMuon'},steps['RECODR2_2016reHLT']])
steps['RECODR2_2017reHLT']=merge([{'--hltProcess':'reHLT','--conditions':'auto:run2_data_relval'},steps['RECODR2_2017']])
steps['RECODR2reHLTAlCaEle']=merge([{'--hltProcess':'reHLT','--conditions':'auto:run2_data_relval'},steps['RECODR2AlCaEle']])
steps['RECODR2reHLTAlCaTkCosmics']=merge([{'--hltProcess':'reHLT','--conditions':'auto:run2_data_relval','-s':'RAW2DIGI,L1Reco,RECO,SKIM:EXONoBPTXSkim,PAT,ALCA:TkAlCosmicsInCollisions,DQM:@standardDQMFakeHLT+@miniAODDQM'},steps['RECODR2_2016']])
steps['RECODR2_2017reHLTAlCaTkCosmics']=merge([{'--hltProcess':'reHLT','--conditions':'auto:run2_data_relval','-s':'RAW2DIGI,L1Reco,RECO,SKIM:EXONoBPTXSkim,PAT,ALCA:TkAlCosmicsInCollisions,DQM:@standardDQMFakeHLT+@miniAODDQM'},steps['RECODR2_2017']])
steps['RECODR2_2017reHLTSiPixelCalZeroBias']=merge([{'--hltProcess':'reHLT','--conditions':'auto:run2_data_relval','-s':'RAW2DIGI,L1Reco,RECO,PAT,ALCA:SiPixelCalZeroBias,DQM:@standardDQMFakeHLT+@miniAODDQM'},steps['RECODR2_2017']])
steps['RECODR2_2018reHLTAlCaTkCosmics']=merge([{'--hltProcess':'reHLT','--conditions':'auto:run2_data_relval','-s':'RAW2DIGI,L1Reco,RECO,SKIM:EXONoBPTXSkim,PAT,ALCA:TkAlCosmicsInCollisions,DQM:@standardDQMFakeHLT+@miniAODDQM'},steps['RECODR2_2018']])

steps['RECODR2_2016reHLT_skimSingleMu']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:ZMu+MuTau,PAT,ALCA:SiPixelCalSingleMuonLoose+SiPixelCalSingleMuonTight+SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT+@muon+@lumi+@L1TMuon'},steps['RECODR2_2016reHLT']])
steps['RECODR2_2016reHLT_skimDoubleEG']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:ZElectron,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT+@ecal+@egamma'},steps['RECODR2_2016reHLT']])
steps['RECODR2_2016reHLT_skimMuonEG']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:TopMuEG,PAT,ALCA:SiPixelCalSingleMuonLoose+SiPixelCalSingleMuonTight+SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT'},steps['RECODR2_2016reHLT']])
steps['RECODR2_2016reHLT_skimJetHT']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:JetHTJetPlusHOFilter,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT+@jetmet+@hcalOnly'},steps['RECODR2_2016reHLT']])
steps['RECODR2_2016reHLT_skimMET']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:HighMET+EXOMONOPOLE,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT+@jetmet+@hcalOnly'},steps['RECODR2_2016reHLT']])
steps['RECODR2_2016reHLT_skimSinglePh']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:SinglePhotonJetPlusHOFilter+EXOMONOPOLE,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT+@ecal+@egamma'},steps['RECODR2_2016reHLT']])
steps['RECODR2_2016reHLT_skimMuOnia']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:BPHSkim,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT+@muon'},steps['RECODR2_2016reHLT']])

steps['RECODR2_2017reHLT_skimSingleMu']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:MuonPOGSkim+ZMu+MuTau,PAT,ALCA:SiPixelCalSingleMuonLoose+SiPixelCalSingleMuonTight+SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT+@muon+@lumi+@L1TMuon'},steps['RECODR2_2017reHLT']])
steps['RECODR2_2017reHLT_skimDoubleEG']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:ZElectron,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT+@ecal+@egamma'},steps['RECODR2_2017reHLT']])
steps['RECODR2_2017reHLT_skimMuonEG']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:TopMuEG,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT'},steps['RECODR2_2017reHLT']])
steps['RECODR2_2017reHLT_skimJetHT']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:JetHTJetPlusHOFilter,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT+@jetmet+@hcalOnly'},steps['RECODR2_2017reHLT']])
steps['RECODR2_2017reHLT_skimDisplacedJet']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:EXODisplacedJet,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT'},steps['RECODR2_2017reHLT']])
steps['RECODR2_2017reHLT_skimMET']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:HighMET+EXOMONOPOLE,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT+@jetmet+@hcalOnly'},steps['RECODR2_2017reHLT']])
steps['RECODR2_2017reHLT_skimSinglePh']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:SinglePhotonJetPlusHOFilter+EXOMONOPOLE,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT+@ecal+@egamma'},steps['RECODR2_2017reHLT']])
steps['RECODR2_2017reHLT_skimMuOnia']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:BPHSkim,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT+@muon'},steps['RECODR2_2017reHLT']])
steps['RECODR2_2017reHLT_skimCharmonium']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:MuonPOGJPsiSkim+BPHSkim,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT'},steps['RECODR2_2017reHLT']])

steps['RECODR2_2018reHLT']=merge([{'--hltProcess':'reHLT','--conditions':'auto:run2_data_relval'},steps['RECODR2_2018']])
steps['RECODR2_2018reHLT_skimSingleMu']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:MuonPOGSkim+ZMu+MuTau,PAT,ALCA:SiPixelCalSingleMuonLoose+SiPixelCalSingleMuonTight+SiStripCalZeroBias+SiStripCalMinBias+SiStripCalSmallBiasScan+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT+@muon+@lumi+@L1TMuon'},steps['RECODR2_2018reHLT']])
steps['RECODR2_2018reHLT_skimEGamma_L1TEgDQM']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:ZElectron+SinglePhotonJetPlusHOFilter+EXOMONOPOLE,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+SiStripCalSmallBiasScan+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT+@ecal+@egamma+@L1TEgamma'},steps['RECODR2_2018reHLT']])
steps['RECODR2_2018reHLT_skimMuonEG']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:TopMuEG,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+SiStripCalSmallBiasScan+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT'},steps['RECODR2_2018reHLT']])
steps['RECODR2_2018reHLT_skimJetHT']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:JetHTJetPlusHOFilter,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+SiStripCalSmallBiasScan+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT+@jetmet+@hcalOnly'},steps['RECODR2_2018reHLT']])
steps['RECODR2_2018reHLT_skimDisplacedJet']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:EXODisplacedJet,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+SiStripCalSmallBiasScan+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT'},steps['RECODR2_2018reHLT']])
steps['RECODR2_2018reHLT_skimMET']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:HighMET+EXOMONOPOLE,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+SiStripCalSmallBiasScan+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT+@jetmet+@hcalOnly'},steps['RECODR2_2018reHLT']])
steps['RECODR2_2018reHLT_skimMuOnia']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:BPHSkim,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+SiStripCalSmallBiasScan+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT+@muon'},steps['RECODR2_2018reHLT']])
steps['RECODR2_2018reHLT_skimCharmonium']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:MuonPOGJPsiSkim+BPHSkim,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+SiStripCalSmallBiasScan+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT+@heavyFlavor'},steps['RECODR2_2018reHLT']])
steps['RECODR2_2018reHLT_skimParkingBPH']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:SkimBPark,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+SiStripCalSmallBiasScan+TkAlMinBias+EcalESAlign,DQM:@commonFakeHLT','--era':'Run2_2018,bParking'},steps['RECODR2_2018reHLT']])

for sname in ['RECODR2_50nsreHLT','RECODR2_50nsreHLT_ZB',
              'RECODR2_25nsreHLT','RECODR2_25nsreHLT_ZB',
              'RECODR2_2016reHLT','RECODR2_2016reHLT_ZB',
              'RECODR2_2016reHLT_L1TEgDQM', 'RECODR2_2016reHLT_L1TMuDQM',
              'RECODR2_2016reHLT_skimDoubleEG', 'RECODR2_2016reHLT_skimJetHT', 'RECODR2_2016reHLT_skimMET',
              'RECODR2_2016reHLT_skimMuonEG', 'RECODR2_2016reHLT_skimSingleMu',
              'RECODR2_2016reHLT_skimSinglePh', 'RECODR2_2016reHLT_skimMuOnia',
              'RECODR2reHLTAlCaTkCosmics']:
    steps[sname+"_HIPM"] = merge([{'--era': steps[sname]['--era']+"_HIPM"},steps[sname]])

# RECO step with offline GT
steps['RECODR2_2016reHLT_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2016reHLT']])
steps['RECODR2_2016reHLT_ZBPrompt']=merge([{'--conditions':'auto:run2_data','-s':'RAW2DIGI,L1Reco,RECO,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@rerecoZeroBiasFakeHLT+@miniAODDQM'},steps['RECODR2_2016reHLT']])
steps['RECODR2_2016reHLT_Prompt_L1TEgDQM']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@standardDQMFakeHLT+@miniAODDQM+@L1TEgamma'},steps['RECODR2_2016reHLT_Prompt']])
steps['RECODR2_2016reHLT_skimDoubleEG_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2016reHLT_skimDoubleEG']])
steps['RECODR2_2016reHLT_skimJetHT_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2016reHLT_skimJetHT']])
steps['RECODR2_2016reHLT_skimMET_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2016reHLT_skimMET']])
steps['RECODR2_2016reHLT_skimMuonEG_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2016reHLT_skimMuonEG']])
steps['RECODR2_2016reHLT_skimSingleMu_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2016reHLT_skimSingleMu']])
steps['RECODR2_2016reHLT_skimSinglePh_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2016reHLT_skimSinglePh']])
steps['RECODR2_2016reHLT_skimMuOnia_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2016reHLT_skimMuOnia']])
steps['RECODR2_2016reHLT_Prompt_Lumi']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@standardDQMFakeHLT+@miniAODDQM+@lumi'},steps['RECODR2_2016reHLT_Prompt']])
steps['RECODR2_2016reHLT_Prompt_Lumi_L1TMuDQM']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@standardDQMFakeHLT+@miniAODDQM+@lumi+@L1TMuon'},steps['RECODR2_2016reHLT_Prompt']])

steps['RECODR2_2017reHLT_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2017reHLT']])
steps['RECODR2_2017reHLT_ZBPrompt']=merge([{'--conditions':'auto:run2_data','-s':'RAW2DIGI,L1Reco,RECO,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@rerecoZeroBiasFakeHLT+@miniAODDQM'},steps['RECODR2_2017reHLT']])
steps['RECODR2_2017reHLT_Prompt_L1TEgDQM']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@standardDQMFakeHLT+@miniAODDQM+@L1TEgamma'},steps['RECODR2_2017reHLT_Prompt']])
steps['RECODR2_2017reHLT_skimDoubleEG_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2017reHLT_skimDoubleEG']])
steps['RECODR2_2017reHLT_skimJetHT_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2017reHLT_skimJetHT']])
steps['RECODR2_2017reHLT_skimDisplacedJet_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2017reHLT_skimDisplacedJet']])
steps['RECODR2_2017reHLT_skimMET_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2017reHLT_skimMET']])
steps['RECODR2_2017reHLT_skimMuonEG_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2017reHLT_skimMuonEG']])
steps['RECODR2_2017reHLT_skimSingleMu_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2017reHLT_skimSingleMu']])
steps['RECODR2_2017reHLT_skimSinglePh_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2017reHLT_skimSinglePh']])
steps['RECODR2_2017reHLT_skimMuOnia_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2017reHLT_skimMuOnia']])
steps['RECODR2_2017reHLT_skimCharmonium_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2017reHLT_skimCharmonium']])
steps['RECODR2_2017reHLT_skimSingleMu_Prompt_Lumi']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:MuonPOGSkim+ZMu+MuTau,PAT,ALCA:SiPixelCalSingleMuonLoose+SiPixelCalSingleMuonTight+SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@standardDQMFakeHLT+@miniAODDQM+@lumi+@L1TMuon'},steps['RECODR2_2017reHLT_skimSingleMu_Prompt']])
steps['RECODR2_2017reHLTAlCaTkCosmics_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2017reHLTAlCaTkCosmics']])
steps['RECODR2_2017reHLTSiPixelCalZeroBias_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2017reHLTSiPixelCalZeroBias']])

steps['RECODR2_2018reHLT_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2018reHLT']])
steps['RECODR2_2018reHLT_ZBPrompt']=merge([{'--conditions':'auto:run2_data','-s':'RAW2DIGI,L1Reco,RECO,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@rerecoZeroBias+@ExtraHLT+@miniAODDQM'},steps['RECODR2_2018reHLT']])
steps['RECODR2_2018reHLT_Prompt_pixelTrackingOnly']=merge([{'-s': 'RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly,DQM:@pixelTrackingOnlyDQM'},steps['RECODR2_2018reHLT_Prompt']])
steps['RECODR2_2018reHLT_Offline']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2018reHLT']])
steps['RECODR2_2018reHLT_ZBOffline']=merge([{'--conditions':'auto:run2_data','-s':'RAW2DIGI,L1Reco,RECO,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@rerecoZeroBias+@ExtraHLT+@miniAODDQM'},steps['RECODR2_2018reHLT']])
steps['RECODR2_2018reHLT_skimEGamma_Prompt_L1TEgDQM']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2018reHLT_skimEGamma_L1TEgDQM']])
steps['RECODR2_2018reHLT_skimEGamma_Offline_L1TEgDQM']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2018reHLT_skimEGamma_L1TEgDQM']])
steps['RECODR2_2018reHLT_skimJetHT_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2018reHLT_skimJetHT']])
steps['RECODR2_2018reHLT_skimJetHT_Offline']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2018reHLT_skimJetHT']])
steps['RECODR2_2018reHLT_skimDisplacedJet_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2018reHLT_skimDisplacedJet']])
steps['RECODR2_2018reHLT_skimDisplacedJet_Offline']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2018reHLT_skimDisplacedJet']])
steps['RECODR2_2018reHLT_skimMET_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2018reHLT_skimMET']])
steps['RECODR2_2018reHLT_skimMET_Offline']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2018reHLT_skimMET']])
steps['RECODR2_2018reHLT_skimMuonEG_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2018reHLT_skimMuonEG']])
steps['RECODR2_2018reHLT_skimMuonEG_Offline']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2018reHLT_skimMuonEG']])
steps['RECODR2_2018reHLT_skimSingleMu_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2018reHLT_skimSingleMu']])
steps['RECODR2_2018reHLT_skimSingleMu_Offline']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2018reHLT_skimSingleMu']])
steps['RECODR2_2018reHLT_skimMuOnia_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2018reHLT_skimMuOnia']])
steps['RECODR2_2018reHLT_skimMuOnia_Offline']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2018reHLT_skimMuOnia']])
steps['RECODR2_2018reHLT_skimCharmonium_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2018reHLT_skimCharmonium']])
steps['RECODR2_2018reHLT_skimCharmonium_Offline']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2018reHLT_skimCharmonium']])
steps['RECODR2_2018reHLT_skimParkingBPH_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2018reHLT_skimParkingBPH']])
steps['RECODR2_2018reHLT_skimParkingBPH_Offline']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2018reHLT_skimParkingBPH']])
steps['RECODR2_2018reHLT_skimSingleMu_Prompt_Lumi']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:MuonPOGSkim+ZMu+MuTau,PAT,ALCA:SiPixelCalSingleMuonLoose+SiPixelCalSingleMuonTight+SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@standardDQMFakeHLT+@miniAODDQM+@lumi+@L1TMuon'},steps['RECODR2_2018reHLT_skimSingleMu_Prompt']])
steps['RECODR2_2018reHLT_skimSingleMu_Offline_Lumi']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:MuonPOGSkim+ZMu+MuTau,PAT,ALCA:SiPixelCalSingleMuonLoose+SiPixelCalSingleMuonTight+SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@standardDQMFakeHLT+@miniAODDQM+@lumi+@L1TMuon'},steps['RECODR2_2018reHLT_skimSingleMu_Offline']])
steps['RECODR2_2018reHLTAlCaTkCosmics_Prompt']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2018reHLTAlCaTkCosmics']])
steps['RECODR2_2018reHLTAlCaTkCosmics_Offline']=merge([{'--conditions':'auto:run2_data'},steps['RECODR2_2018reHLTAlCaTkCosmics']])
steps['RECODR2_2018reHLT_Prompt_hBStar']=merge([{'--era':'Run2_2018_highBetaStar'},steps['RECODR2_2018reHLT_Prompt']])
steps['RECODR2_2018reHLT_Offline_hBStar']=merge([{'--era':'Run2_2018_highBetaStar'},steps['RECODR2_2018reHLT_Offline']])
steps['RECODR2_2018reHLT_skimJetHT_Prompt_HEfail']=merge([{'--conditions':'auto:run2_data_HEfail'},steps['RECODR2_2018reHLT_skimJetHT']])
steps['RECO']=merge([step3Defaults])


steps['RECOAlCaCalo']=merge([step3DefaultsAlCaCalo])
steps['RECODBG']=merge([{'--eventcontent':'RECODEBUG,MINIAODSIM,DQM'},steps['RECO']])
steps['RECOPROD1']=merge([{ '-s' : 'RAW2DIGI,L1Reco,RECO,RECOSIM,PAT', '--datatier' : 'AODSIM,MINIAODSIM', '--eventcontent' : 'AODSIM,MINIAODSIM'},step3Defaults])
#steps['RECOPRODUP15']=merge([{ '-s':'RAW2DIGI,L1Reco,RECO,DQM:DQMOfflinePOGMC','--datatier':'AODSIM,DQMIO','--eventcontent':'AODSIM,DQM'},step3Up2015Defaults])
steps['RECOPRODUP15']=merge([{ '-s':'RAW2DIGI,L1Reco,RECO,RECOSIM,PAT','--datatier':'AODSIM,MINIAODSIM','--eventcontent':'AODSIM,MINIAODSIM'},step3Up2015Defaults])
## for 2017 PROD
steps['RECOPRODUP17']=merge([{ '--era' :'Run2_2017','--conditions': 'auto:phase1_2017_realistic'},steps['RECOPRODUP15']])
## for 2018 PROD
steps['RECOPRODUP18']=merge([{ '-s':'RAW2DIGI,L1Reco,RECO,RECOSIM','--era' :'Run2_2018','--conditions': 'auto:phase1_2018_realistic','--datatier':'AODSIM','--eventcontent':'AODSIM'},step3Up2015Defaults])
steps['RECOPRODUP18bParking']=merge([{ '-s':'RAW2DIGI,L1Reco,RECO,RECOSIM','--era' :'Run2_2018,bParking','--conditions': 'auto:phase1_2018_realistic','--datatier':'AODSIM','--eventcontent':'AODSIM'},step3Up2015Defaults])
##
steps['RECOCOS']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,DQM','--scenario':'cosmics','--datatier':'GEN-SIM-RECO,DQMIO','--eventcontent':'RECOSIM,DQM'},stCond,step3Defaults])
steps['RECOHAL']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,DQM','--scenario':'cosmics'},step3Up2015Hal])
steps['RECOCOS_UP15']=merge([{'--conditions':'auto:run2_mc_cosmics','-s':'RAW2DIGI,L1Reco,RECO,ALCA:MuAlGlobalCosmics,DQM','--scenario':'cosmics'},step3Up2015Hal])
steps['RECOCOS_UP16']=merge([{'--conditions':'auto:run2_mc_cosmics','-s':'RAW2DIGI,L1Reco,RECO,ALCA:MuAlGlobalCosmics,DQM','--scenario':'cosmics','--era':'Run2_2016'},step3Up2015Hal])
steps['RECOCOS_UP17']=merge([{'--conditions':'auto:phase1_2017_cosmics','-s':'RAW2DIGI,L1Reco,RECO,ALCA:MuAlGlobalCosmics,DQM','--scenario':'cosmics','--era':'Run2_2017'},step3Up2015Hal])
steps['RECOCOS_UP18']=merge([{'--conditions':'auto:phase1_2018_cosmics','-s':'RAW2DIGI,L1Reco,RECO,ALCA:MuAlGlobalCosmics,DQM','--scenario':'cosmics','--era':'Run2_2018'},step3Up2015Hal])
steps['RECOCOS_UP21']=merge([{'--conditions':'auto:phase1_2022_cosmics','-s':'RAW2DIGI,L1Reco,RECO,ALCA:MuAlGlobalCosmics,DQM','--scenario':'cosmics','--era':'Run3'},step3Up2015Hal])
steps['RECOCOS_UP21_0T']=merge([{'--magField':'0T','--conditions':'auto:phase1_2022_cosmics_0T'},steps['RECOCOS_UP21']])
steps['RECOCOSPEAK_UP17']=merge([{'--conditions':'auto:phase1_2017_cosmics_peak','-s':'RAW2DIGI,L1Reco,RECO,ALCA:MuAlGlobalCosmics,DQM','--scenario':'cosmics','--era':'Run2_2017'},step3Up2015Hal])
steps['RECOCOSPEAK_UP18']=merge([{'--conditions':'auto:phase1_2018_cosmics_peak','-s':'RAW2DIGI,L1Reco,RECO,ALCA:MuAlGlobalCosmics,DQM','--scenario':'cosmics','--era':'Run2_2018'},step3Up2015Hal])


steps['RECOMIN']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,RECOSIM,ALCA:SiStripCalZeroBias+SiStripCalMinBias,VALIDATION,DQM'},stCond,step3Defaults])
steps['RECOMINUP15']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,RECOSIM,ALCA:SiStripCalZeroBias+SiStripCalMinBias,VALIDATION,DQM'},step3Up2015Defaults])
steps['RECOAODUP15']=merge([{'--datatier':'AODSIM,MINIAODSIM,DQMIO','--eventcontent':'AODSIM,MINIAODSIM,DQM'},step3Up2015Defaults])

steps['RECODDQM']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,DQM:@allForPrompt'},steps['RECOD']])

steps['RECOPU1']=merge([PU,steps['RECO']])
steps['RECOPU2']=merge([PU2,steps['RECO']])
steps['RECOUP15_PU25']=merge([PU25,step3Up2015Defaults])
steps['RECOUP15_PU25_HIPM']=merge([step3_HIPM,steps['RECOUP15_PU25']])
steps['RECOUP15_PU25_L1TEgDQM']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM+@L1TEgamma'},steps['RECOUP15_PU25']])
steps['RECOUP15_PU25_L1TMuDQM']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM+@L1TMuon'},steps['RECOUP15_PU25']])
steps['RECOUP15_PU50']=merge([PU50,step3Up2015Defaults50ns])
steps['RECOUP15_PU50_L1TEgDQM']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM+@L1TEgamma'},steps['RECOUP15_PU50']])
steps['RECOUP15_PU50_L1TMuDQM']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM+@L1TMuon'},steps['RECOUP15_PU50']])

# for PU25 High stats workflows
steps['RECOUP15_PU25HS']=merge([PU25HS,step3Up2015Defaults])

# Run-3 reco
steps['RECODR3_reHLT_MinBiasOffline']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,PAT,ALCA:SiStripCalMinBias+TkAlMinBias+EcalESAlign,DQM:@commonSiStripZeroBias+@miniAODDQM','--procModifiers':'siPixelQualityRawToDigi'},steps['RECODR3_reHLT_2022']])
steps['RECODR3_reHLT_ZBOffline']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,PAT,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+EcalESAlign+HcalCalIsoTrkProducerFilter,DQM:@rerecoZeroBias+@miniAODDQM','--procModifiers':'siPixelQualityRawToDigi'},steps['RECODR3_reHLT_2022']])
steps['RECODR3_reHLT_HLTPhysics_Offline']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,PAT,ALCA:TkAlMinBias+HcalCalIterativePhiSym+HcalCalIsoTrkProducerFilter+HcalCalHO+HcalCalHBHEMuonProducerFilter,DQM:@commonReduced+@miniAODDQM','--procModifiers':'siPixelQualityRawToDigi'},steps['RECODR3_reHLT_2022']])
steps['RECODR3_reHLT_AlCaTkCosmics_Offline']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:EXONoBPTXSkim,PAT,ALCA:TkAlCosmicsInCollisions,DQM:@standardDQMFakeHLT+@miniAODDQM'},steps['RECODR3_reHLT_2022']])
steps['RECODR3_reHLT_pixelTrackingOnly']=merge([{'-s': 'RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly,DQM:@pixelTrackingOnlyDQM'},steps['RECODR3_reHLT_2023']])
steps['RECODR3_reHLT_Patatrack_PixelOnlyCPU']=merge([step3_pixel_ntuplet_cpu, steps['RECODR3_reHLT_pixelTrackingOnly']])
steps['RECODR3_reHLT_Patatrack_PixelOnlyGPU']=merge([step3_pixel_ntuplet_gpu, steps['RECODR3_reHLT_pixelTrackingOnly']])
steps['RECODR3_reHLT_Patatrack_PixelOnlyGPUValidation']=merge([{'--accelerators':'gpu-nvidia'},step3_pixel_ntuplet_gpu_validation,steps['RECODR3_reHLT_Patatrack_PixelOnlyGPU']])
steps['RECODR3_reHLT_Patatrack_PixelOnlyGPUProfiling']=merge([{'-s':'RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly','--customise':'RecoTracker/Configuration/customizePixelOnlyForProfiling.customizePixelOnlyForProfilingGPUOnly'},steps['RECODR3_reHLT_Patatrack_PixelOnlyGPU']])
steps['RECODR3_reHLT_Patatrack_PixelOnlyTripletsCPU']=merge([step3_pixel_ntuplet_cpu, step3_pixel_triplets, steps['RECODR3_reHLT_pixelTrackingOnly']])
steps['RECODR3_reHLT_Patatrack_PixelOnlyTripletsGPU']=merge([step3_pixel_ntuplet_gpu, step3_pixel_triplets, steps['RECODR3_reHLT_pixelTrackingOnly']])
steps['RECODR3_reHLT_Patatrack_PixelOnlyTripletsGPUValidation']=merge([{'--accelerators':'gpu-nvidia'},step3_pixel_ntuplet_gpu_validation,steps['RECODR3_reHLT_Patatrack_PixelOnlyTripletsGPU']])
steps['RECODR3_reHLT_Patatrack_PixelOnlyTripletsGPUProfiling']=merge([{'-s':'RAW2DIGI:RawToDigi_pixelOnly,RECO:reconstruction_pixelTrackingOnly','--customise':'RecoTracker/Configuration/customizePixelTracksForTriplets.customizePixelTracksForTriplets,RecoTracker/Configuration/customizePixelOnlyForProfiling.customizePixelOnlyForProfilingGPUOnly'},steps['RECODR3_reHLT_Patatrack_PixelOnlyTripletsGPU']])
steps['RECODR3_reHLT_ECALOnlyCPU']=merge([{'-s': 'RAW2DIGI:RawToDigi_ecalOnly,RECO:reconstruction_ecalOnly,DQM:@ecalOnly'},steps['RECODR3_reHLT_2023']])
steps['RECODR3_reHLT_ECALOnlyGPU']=merge([step3_gpu, steps['RECODR3_reHLT_ECALOnlyCPU']])
steps['RECODR3_reHLT_ECALOnlyGPUValidation']=merge([{'--accelerators':'gpu-nvidia','--procModifiers':'gpuValidation'},steps['RECODR3_reHLT_ECALOnlyGPU']])
steps['RECODR3_reHLT_ECALOnlyGPUProfiling']=merge([{'-s':'RAW2DIGI:RawToDigi_ecalOnly,RECO:reconstruction_ecalOnly', '--customise':'RecoLocalCalo/Configuration/customizeEcalOnlyForProfiling.customizeEcalOnlyForProfilingGPUOnly'},steps['RECODR3_reHLT_ECALOnlyGPU']])
steps['RECODR3_reHLT_HCALOnlyCPU']=merge([{'-s': 'RAW2DIGI:RawToDigi_hcalOnly,RECO:reconstruction_hcalOnly,DQM:@hcalOnly+@hcal2Only'},steps['RECODR3_reHLT_2023']])
steps['RECODR3_reHLT_HCALOnlyGPU']=merge([step3_gpu, steps['RECODR3_reHLT_HCALOnlyCPU']])
steps['RECODR3_reHLT_HCALOnlyGPUValidation']=merge([{'--accelerators':'gpu-nvidia','--procModifiers':'gpuValidation'},steps['RECODR3_reHLT_HCALOnlyGPU']])
steps['RECODR3_reHLT_HCALOnlyGPUProfiling']=merge([{'-s':'RAW2DIGI:RawToDigi_hcalOnly,RECO:reconstruction_hcalOnly','--customise':'RecoLocalCalo/Configuration/customizeHcalOnlyForProfiling.customizeHcalOnlyForProfilingGPUOnly'},steps['RECODR3_reHLT_HCALOnlyGPU']])

steps['RECONANORUN3_reHLT_2022']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,PAT,NANO,DQM:@standardDQMFakeHLT+@miniAODDQM+@nanoAODDQM','--datatier':'RECO,MINIAOD,NANOAOD,DQMIO','--eventcontent':'RECO,MINIAOD,NANOEDMAOD,DQM'},steps['RECODR3_reHLT_2022']])
steps['RECONANORUN3_ZB_reHLT_2022']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,PAT,NANO,DQM:@rerecoZeroBiasFakeHLT+@miniAODDQM+@nanoAODDQM'},steps['RECONANORUN3_reHLT_2022']])
steps['RECOCOSMRUN3_reHLT_2022']=merge([{'--scenario':'cosmics','-s':'RAW2DIGI,L1Reco,RECO,DQM','--datatier':'RECO,DQMIO','--eventcontent':'RECO,DQM'},steps['RECONANORUN3_reHLT_2022']])

steps['RECONANORUN3_reHLT_2023']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,PAT,NANO,DQM:@standardDQM+@miniAODDQM+@nanoAODDQM','--datatier':'RECO,MINIAOD,NANOAOD,DQMIO','--eventcontent':'RECO,MINIAOD,NANOEDMAOD,DQM'},steps['RECODR3_reHLT_2023']])
steps['RECONANORUN3_reHLT_2023B']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,PAT,NANO,DQM:@standardDQM+@miniAODDQM+@nanoAODDQM','--datatier':'RECO,MINIAOD,NANOAOD,DQMIO','--eventcontent':'RECO,MINIAOD,NANOEDMAOD,DQM'},steps['RECODR3_reHLT_2023B']])
steps['RECONANORUN3_ZB_reHLT_2023B']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,PAT,NANO,DQM:@rerecoZeroBias+@miniAODDQM+@nanoAODDQM'},steps['RECONANORUN3_reHLT_2023B']])
steps['RECONANORUN3_ZB_reHLT_2023']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,PAT,NANO,DQM:@rerecoZeroBias+@miniAODDQM+@nanoAODDQM'},steps['RECONANORUN3_reHLT_2023']])
steps['RECOCOSMRUN3_reHLT_2023']=merge([{'--scenario':'cosmics','-s':'RAW2DIGI,L1Reco,RECO,DQM','--datatier':'RECO,DQMIO','--eventcontent':'RECO,DQM'},steps['RECONANORUN3_reHLT_2023']])

# Run-3 2022 skim
for s in autoSkim.keys():

    if 'ZeroBias' in s:
        steps['SKIM'+s.upper()+'RUN3_reHLT_2022'] = merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:%s,PAT,NANO,DQM:@rerecoZeroBiasFakeHLT+@miniAODDQM+@nanoAODDQM'%(autoSkim[s])},steps['RECONANORUN3_reHLT_2022']])
    elif 'Cosmics' in s:
        steps['SKIM'+s.upper()+'RUN3_reHLT_2022'] = merge([{'--scenario':'cosmics', '-s':'RAW2DIGI,L1Reco,RECO,SKIM:%s,DQM'%(autoSkim[s]), '--datatier':'RECO,DQMIO', '--eventcontent':'RECO,DQM'},steps['RECONANORUN3_reHLT_2022']])
    else:
        steps['SKIM'+s.upper()+'RUN3_reHLT_2022'] = merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:%s,PAT,NANO,DQM:@standardDQMFakeHLT+@miniAODDQM+@nanoAODDQM'%(autoSkim[s])},steps['RECONANORUN3_reHLT_2022']])

# Run-3 2023 skim
for s in autoSkim.keys():

    if 'ZeroBias' in s:
        steps['SKIM'+s.upper()+'RUN3_reHLT_2023'] = merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:%s,PAT,NANO,DQM:@rerecoZeroBias+@miniAODDQM+@nanoAODDQM'%(autoSkim[s])},steps['RECONANORUN3_reHLT_2023']])
    elif 'Cosmics' in s:
        steps['SKIM'+s.upper()+'RUN3_reHLT_2023'] = merge([{'--scenario':'cosmics', '-s':'RAW2DIGI,L1Reco,RECO,SKIM:%s,DQM'%(autoSkim[s]), '--datatier':'RECO,DQMIO', '--eventcontent':'RECO,DQM'},steps['RECONANORUN3_reHLT_2023']])
    else:
        steps['SKIM'+s.upper()+'RUN3_reHLT_2023'] = merge([{'-s':'RAW2DIGI,L1Reco,RECO,SKIM:%s,PAT,NANO,DQM:@standardDQM+@miniAODDQM+@nanoAODDQM'%(autoSkim[s])},steps['RECONANORUN3_reHLT_2023']])


# mask away - to be removed once we'll migrate the matrix to be fully unscheduled for RECO step
#steps['RECOmAOD']=merge([step3DefaultsUnsch])
#steps['RECOmAODUP15']=merge([step3Up2015DefaultsUnsch])


# for premixing: no --pileup_input for replay; GEN-SIM only available for in-time event, from FEVTDEBUGHLT previous step
steps['RECOPRMXUP15_PU25']=merge([
        {'--era':'Run2_2016','--procModifiers':'premix_stage2'}, # temporary replacement for premix; to be brought back to customisePostLS1; DataMixer customize for rerouting inputs to mixed data.
        step3Up2015Defaults])
steps['RECOPRMXUP15_HIPM_PU25']=merge([step3_HIPM,steps['RECOPRMXUP15_PU25']])
steps['RECOPRMXUP15_PU50']=merge([
        {'--era':'Run2_50ns','--procModifiers':'premix_stage2'},
        step3Up2015Defaults50ns])

steps['RECOPRMXUP17_PU25']=merge([
        {'--conditions':'auto:phase1_2017_realistic','--era':'Run2_2017','--procModifiers':'premix_stage2'},
        step3Up2015Defaults])

steps['RECOPRMXUP18_PU25']=merge([
        {'--conditions':'auto:phase1_2018_realistic','--era':'Run2_2018','--procModifiers':'premix_stage2'},
        step3Up2015Defaults])
steps['RECOPRMXUP18_PU25_L1TEgDQM']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM+@L1TEgamma'},steps['RECOPRMXUP18_PU25']])
steps['RECOPRMXUP18_PU25_L1TMuDQM']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM+@L1TMuon'},steps['RECOPRMXUP18_PU25']])

#Run-Dependent RECO
step_RECO18_RD = {'--conditions':'auto:phase1_2018_realistic_rd'}
steps['RECOPRMXUP18_PU25_RD']=merge([step_RECO18_RD, steps['RECOPRMXUP18_PU25']])
steps['RECOPRMXUP18_PU25_L1TEgDQM_RD']=merge([step_RECO18_RD, steps['RECOPRMXUP18_PU25_L1TEgDQM']])
steps['RECOPRMXUP18_PU25_L1TMuDQM_RD']=merge([step_RECO18_RD, steps['RECOPRMXUP18_PU25_L1TMuDQM']])

recoPremixUp15prod = merge([
        #{'-s':'RAW2DIGI,L1Reco,RECO'}, # tmp
        {'-s':'RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,DQM:DQMOfflinePOGMC'},
        {'--datatier' : 'AODSIM,MINIAODSIM,DQMIO'},
        {'--eventcontent' : 'AODSIM,MINIAODSIM,DQMIO'},
        {'--era':'Run2_2016'}, # temporary replacement for premix; to be brought back to customisePostLS1
        step3Up2015Defaults])

steps['RECOPRMXUP15PROD_PU25']=merge([
        recoPremixUp15prod])
steps['RECOPRMXUP15PROD_PU50']=merge([
        {'--conditions':'auto:run2_mc_50ns'},
        {'--era':'Run2_50ns'},
        recoPremixUp15prod])

recoPremixUp17prod = merge([
        {'-s':'RAW2DIGI,L1Reco,RECO,RECOSIM,PAT'},
        {'--datatier' : 'AODSIM,MINIAODSIM'},
        {'--eventcontent' : 'AODSIM,MINIAODSIM'},
        {'--era':'Run2_2017'},
        {'--conditions':'auto:phase1_2017_realistic'},
        step3Up2015Defaults])


steps['RECOPRMXUP17PROD_PU25']=merge([recoPremixUp17prod])
steps['RECOPRMXUP18PROD_PU25']=merge([{'--era':'Run2_2018','--conditions':'auto:phase1_2018_realistic','--procModifiers':'premix_stage2'},recoPremixUp17prod])


steps['RECOPUDBG']=merge([{'--eventcontent':'RECODEBUG,MINIAODSIM,DQM'},steps['RECOPU1']])
steps['RERECOPU1']=merge([{'--hltProcess':'REDIGI'},steps['RECOPU1']])

steps['RECOUP15_ID']=merge([{'--hltProcess':'HLT2'},steps['RECOUP15']])

steps['RECOHI2023PPRECOAPPROXCLUSTERS']=merge([hiDefaults2023_ppReco_approxClusters,{'-s':'RAW2DIGI,L1Reco,RECO,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM',
                                                        '--datatier':'GEN-SIM-RECO,MINIAODSIM,DQMIO',
                                                        '--eventcontent':'RECOSIM,MINIAODSIM,DQM',
                                                        },step3Up2015Defaults])
steps['RECOHI2023PPRECOMBAPPROXCLUSTERS']=merge([hiDefaults2023_ppReco_approxClusters,{'-s':'RAW2DIGI,L1Reco,RECO,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM',
                                                          '--datatier':'GEN-SIM-RECO,MINIAODSIM,DQMIO',
                                                          '--eventcontent':'RECOSIM,MINIAODSIM,DQM',
                                                          '--era':'Run3_pp_on_PbPb_2023',
                                                          '--procModifiers':'genJetSubEvent',
                                                        },step3Up2015Defaults])

steps['RECOHI2023PROD']=merge([hiDefaults2023_ppReco_approxClusters,{'-s':'RAW2DIGI,L1Reco,RECO',
                                                                     '--datatier':'AODSIM',
                                                                     '--eventcontent':'AODSIM',
                                                                     '--era':'Run3_pp_on_PbPb_2023',
                                                        },step3Up2015Defaults])

steps['MINIHI2023PROD']=merge([hiDefaults2023_ppReco,{'-s':'PAT',
                                                                     '--datatier':'MINIAODSIM',
                                                                     '--eventcontent':'MINIAODSIM',
                                                                     '--era':'Run3_pp_on_PbPb_2023',
                                                                     '--procModifiers':'genJetSubEvent',
                                                        },step3Up2015Defaults])

steps['RECOHI2023PPRECO']=merge([hiDefaults2023_ppReco,{'-s':'RAW2DIGI,L1Reco,RECO,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM',
                                                        '--datatier':'GEN-SIM-RECO,MINIAODSIM,DQMIO',
                                                        '--eventcontent':'RECOSIM,MINIAODSIM,DQM',
                                                        },step3Up2015Defaults])
steps['RECOHI2023PPRECOMB']=merge([hiDefaults2023_ppReco,{'-s':'RAW2DIGI,L1Reco,RECO,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM',
                                                          '--datatier':'GEN-SIM-RECO,MINIAODSIM,DQMIO',
                                                          '--eventcontent':'RECOSIM,MINIAODSIM,DQM',
                                                          '--era':'Run3_pp_on_PbPb_2023',
                                                          '--procModifiers':'genJetSubEvent',
                                                        },step3Up2015Defaults])
steps['RECOHI2023PPRECOMB_PatatrackGPU']=merge([hiDefaults2023_ppReco,step3_pixel_ntuplet_gpu,{'-s':'RAW2DIGI,L1Reco,RECO,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM',
                                                          '--datatier':'GEN-SIM-RECO,MINIAODSIM,DQMIO',
                                                          '--eventcontent':'RECOSIM,MINIAODSIM,DQM',
                                                          '--era':'Run3_pp_on_PbPb',
                                                          '--procModifiers':'genJetSubEvent',
                                                        },step3Up2015Defaults])
steps['RECOHI2023PPRECOMB_PatatrackCPU']=merge([hiDefaults2023_ppReco,step3_pixel_ntuplet_cpu,{'-s':'RAW2DIGI,L1Reco,RECO,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM',
                                                          '--datatier':'GEN-SIM-RECO,MINIAODSIM,DQMIO',
                                                          '--eventcontent':'RECOSIM,MINIAODSIM,DQM',
                                                          '--era':'Run3_pp_on_PbPb',
                                                          '--procModifiers':'genJetSubEvent',
                                                        },step3Up2015Defaults])
steps['REMINIAODHI2023PPRECOMB']=merge([{'-s':'PAT,VALIDATION:@miniAODValidation,DQM:@miniAODDQM',
                                         '--datatier':'MINIAODSIM,DQMIO',
                                         '--eventcontent':'MINIAODSIM,DQM',
                                         '-n':100,
                                         '--era':'Run3_pp_on_PbPb_2023',
                                         '--procModifiers':'genJetSubEvent',
                                     },hiDefaults2023_ppReco,step3Up2015Defaults])

steps['ALCARECOHI2023PPRECO']=merge([hiDefaults2023_ppReco,{'-s':'ALCA:TkAlMinBias+SiStripCalMinBias',
                                                            '--datatier':'ALCARECO',
                                                            '--eventcontent':'ALCARECO'
                                                            }])

steps['RECOHI2022PPRECOAPPROXCLUSTERS']=merge([hiDefaults2022_ppReco_approxClusters,{'-s':'RAW2DIGI,L1Reco,RECO,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM',
                                                        '--datatier':'GEN-SIM-RECO,MINIAODSIM,DQMIO',
                                                        '--eventcontent':'RECOSIM,MINIAODSIM,DQM',
                                                        },step3Up2015Defaults])
steps['RECOHI2022PPRECOMBAPPROXCLUSTERS']=merge([hiDefaults2022_ppReco_approxClusters,{'-s':'RAW2DIGI,L1Reco,RECO,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM',
                                                          '--datatier':'GEN-SIM-RECO,MINIAODSIM,DQMIO',
                                                          '--eventcontent':'RECOSIM,MINIAODSIM,DQM',
                                                          '--era':'Run3_pp_on_PbPb',
                                                          '--procModifiers':'genJetSubEvent',
                                                        },step3Up2015Defaults])

steps['RECOHI2022PROD']=merge([hiDefaults2022_ppReco_approxClusters,{'-s':'RAW2DIGI,L1Reco,RECO',
                                                                     '--datatier':'AODSIM',
                                                                     '--eventcontent':'AODSIM',
                                                                     '--era':'Run3_pp_on_PbPb',
                                                        },step3Up2015Defaults])

steps['MINIHI2022PROD']=merge([hiDefaults2022_ppReco,{'-s':'PAT',
                                                                     '--datatier':'MINIAODSIM',
                                                                     '--eventcontent':'MINIAODSIM',
                                                                     '--era':'Run3_pp_on_PbPb',
                                                                     '--procModifiers':'genJetSubEvent',
                                                        },step3Up2015Defaults])

steps['RECOHI2022PPRECO']=merge([hiDefaults2022_ppReco,{'-s':'RAW2DIGI,L1Reco,RECO,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM',
                                                        '--datatier':'GEN-SIM-RECO,MINIAODSIM,DQMIO',
                                                        '--eventcontent':'RECOSIM,MINIAODSIM,DQM',
                                                        },step3Up2015Defaults])
steps['RECOHI2022PPRECOMB']=merge([hiDefaults2022_ppReco,{'-s':'RAW2DIGI,L1Reco,RECO,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM',
                                                          '--datatier':'GEN-SIM-RECO,MINIAODSIM,DQMIO',
                                                          '--eventcontent':'RECOSIM,MINIAODSIM,DQM',
                                                          '--era':'Run3_pp_on_PbPb',
                                                          '--procModifiers':'genJetSubEvent',
                                                        },step3Up2015Defaults])
steps['REMINIAODHI2022PPRECOMB']=merge([{'-s':'PAT,VALIDATION:@miniAODValidation,DQM:@miniAODDQM',
                                         '--datatier':'MINIAODSIM,DQMIO',
                                         '--eventcontent':'MINIAODSIM,DQM',
                                         '-n':100,
                                         '--era':'Run3_pp_on_PbPb',
                                         '--procModifiers':'genJetSubEvent',
                                     },hiDefaults2022_ppReco,step3Up2015Defaults])

steps['ALCARECOHI2022PPRECO']=merge([hiDefaults2022_ppReco,{'-s':'ALCA:TkAlMinBias+SiStripCalMinBias',
                                                            '--datatier':'ALCARECO',
                                                            '--eventcontent':'ALCARECO'
                                                            }])

steps['RECOHI2018PPRECO']=merge([hiDefaults2018_ppReco,{'-s':'RAW2DIGI,L1Reco,RECO,ALCA:SiStripCalZeroBias+SiPixelCalZeroBias,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM',
                                                        '--datatier':'GEN-SIM-RECO,MINIAODSIM,DQMIO,ALCARECO',
                                                        '--eventcontent':'RECOSIM,MINIAODSIM,DQM,ALCARECO',
                                                        },step3Up2015Defaults])
steps['RECOHI2018PPRECOMB']=merge([hiDefaults2018_ppReco,{'-s':'RAW2DIGI,L1Reco,RECO,ALCA:SiStripCalZeroBias+SiPixelCalZeroBias,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM',
                                                          '--datatier':'GEN-SIM-RECO,MINIAODSIM,DQMIO,ALCARECO',
                                                          '--eventcontent':'RECOSIM,MINIAODSIM,DQM,ALCARECO',
                                                          '--era':'Run2_2018_pp_on_AA',
                                                          '--procModifiers':'genJetSubEvent',
                                                      },step3Up2015Defaults])

steps['RECOHI2023MIX']=merge([hiDefaults2023_ppReco,{'-s':'RAW2DIGI,L1Reco,RECO,PAT,VALIDATION:@standardValidationNoHLTHiMix+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM',
                                                     '--pileup':'HiMix',
                                                     '--pileup_input':'das:/RelValHydjetQ_MinBias_5362GeV_2023_ppReco/%s/GEN-SIM'%(baseDataSetRelease[25])
                                                 },step3Up2015Defaults])

steps['RECOHI2022MIX']=merge([hiDefaults2022_ppReco,{'-s':'RAW2DIGI,L1Reco,RECO,PAT,VALIDATION:@standardValidationNoHLTHiMix+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM',
                                                     '--pileup':'HiMix',
                                                     '--pileup_input':'das:/RelValHydjetQ_B12_5020GeV_2021_ppReco/%s/GEN-SIM'%(baseDataSetRelease[23])
                                                 },step3Up2015Defaults])


steps['RECOHIMIX']=merge([hiDefaults2018_ppReco,{'-s':'RAW2DIGI,L1Reco,RECO,PAT,VALIDATION:@standardValidationNoHLTHiMix+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM',
                                                 '--era':'Run2_2018_pp_on_AA',
                                                 '--pileup':'HiMix',
                                                 '--pileup_input':'das:/RelValHydjetQ_B12_5020GeV_2018/%s/GEN-SIM'%(baseDataSetRelease[9])
                                             },step3Up2015Defaults])


steps['REMINIAODHI2018PPRECO']=merge([{'-s':'PAT,VALIDATION:@miniAODValidation,DQM:@miniAODDQM',
                                       '--datatier':'MINIAODSIM,DQMIO',
                                       '--eventcontent':'MINIAODSIM,DQM',
                                       '-n':100,
                                       '--era':'Run2_2018_pp_on_AA',
                                       '--procModifiers':'run2_miniAOD_pp_on_AA_103X',
                                   },hiDefaults2018_ppReco,step3Up2015Defaults])
steps['REMINIAODHI2018PPRECOMB']=merge([{'-s':'PAT,VALIDATION:@miniAODValidation,DQM:@miniAODDQM',
                                         '--datatier':'MINIAODSIM,DQMIO',
                                         '--eventcontent':'MINIAODSIM,DQM',
                                         '-n':100,
                                         '--era':'Run2_2018_pp_on_AA',
                                         '--procModifiers':'genJetSubEvent,run2_miniAOD_pp_on_AA_103X',
                                     },hiDefaults2018_ppReco,step3Up2015Defaults])
steps['ALCARECOHI2018PPRECO']=merge([hiDefaults2018_ppReco,{'-s':'ALCA:TkAlMinBias+SiStripCalMinBias',
                                                            '--datatier':'ALCARECO',
                                                            '--eventcontent':'ALCARECO'
                                                            }])

steps['RECOHI2018']=merge([hiDefaults2018,{'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM'},step3Up2015Defaults])
steps['RECOHI2017']=merge([hiDefaults2017,{'-s':'RAW2DIGI,L1Reco,RECO,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM'},step3Up2015Defaults])
steps['RECOHI2015']=merge([hiDefaults2015,{'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM'},step3Up2015Defaults])
steps['RECOHI2011']=merge([hiDefaults2011,{'-s':'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM'},step3Defaults])

steps['RECOPPREF2017']=merge([ppRefDefaults2017,step3Up2015Defaults])

steps['RECOHID11St3']=merge([{
                              '--process':'ZStoRECO'},
                             steps['RECOHID11']])
steps['RECOHIR10D11']=merge([{'--filein':'file:step2_inREPACKRAW.root',
                              '--filtername':'reRECO'},
                             steps['RECOHID11St3']])
#steps['RECOFS']=merge([{'--fast':'',
#                        '-s':'RECO,HLT:@fake,VALIDATION'},
#                       steps['RECO']])

#add this line when testing from an input file that is not strictly GEN-SIM
#addForAll(step3,{'--hltProcess':'DIGI'})

steps['ALCACOSD']={'--conditions':'auto:run1_data',
                   '--datatier':'ALCARECO',
                   '--eventcontent':'ALCARECO',
                   '--scenario':'cosmics',
                   '-s':'ALCA:TkAlCosmics0T+MuAlGlobalCosmics+HcalCalHOCosmics+DQM'
                   }

steps['ALCACOSDRUN2']=merge([{'--conditions':'auto:run2_data','--era':'Run2_2016','-s':'ALCA:SiPixelCalCosmics+TkAlCosmics0T+SiStripCalCosmics+SiStripCalCosmicsNano+MuAlGlobalCosmics+HcalCalHOCosmics+DQM'},steps['ALCACOSD']])

steps['ALCACOSDRUN3']=merge([{'--conditions':'auto:run3_data','--era':'Run3','-s':'ALCA:SiPixelCalCosmics+TkAlCosmics0T+SiStripCalCosmics+SiStripCalCosmicsNano+MuAlGlobalCosmics+HcalCalHOCosmics+DQM'},steps['ALCACOSD']])
steps['ALCACOSDPROMPTRUN3']=merge([{'--conditions':'auto:run3_data_prompt','--era':'Run3','-s':'ALCA:SiPixelCalCosmics+TkAlCosmics0T+SiStripCalCosmics+SiStripCalCosmicsNano+MuAlGlobalCosmics+HcalCalHOCosmics+DQM'},steps['ALCACOSD']])
steps['ALCACOSDEXPRUN3']=merge([{'--conditions':'auto:run3_data_express','--era':'Run3','-s':'ALCA:SiPixelCalCosmics+TkAlCosmics0T+SiStripCalCosmics+SiStripCalCosmicsNano+MuAlGlobalCosmics+HcalCalHOCosmics+DQM'},steps['ALCACOSD']])

steps['ALCAPROMPT']={'-s':'ALCA:PromptCalibProd',
                     '--filein':'file:TkAlMinBias.root',
                     '--conditions':'auto:run1_data',
                     '--datatier':'ALCARECO',
                     '--eventcontent':'ALCARECO'}
steps['ALCAEXP']={'-s':'ALCAOUTPUT:SiStripCalZeroBias+TkAlMinBias+Hotline+LumiPixelsMinBias+AlCaPCCZeroBiasFromRECO+AlCaPCCRandomFromRECO+SiPixelCalSingleMuon,ALCA:PromptCalibProd+PromptCalibProdSiStrip+PromptCalibProdSiStripGains+PromptCalibProdSiStripGainsAAG+PromptCalibProdSiStripHitEff+PromptCalibProdSiPixelAli+PromptCalibProdSiPixel+PromptCalibProdSiPixelLA',
                  '--customise': 'Alignment/CommonAlignmentProducer/customizeLSNumberFilterForRelVals.doNotFilterLS',
                  '--conditions':'auto:run1_data',
                  '--datatier':'ALCARECO',
                  '--eventcontent':'ALCARECO',
                  '--triggerResultsProcess': 'RECO'}

steps['ALCAEXPRUN2']={'-s':'ALCAOUTPUT:SiStripCalZeroBias+TkAlMinBias+LumiPixelsMinBias+AlCaPCCZeroBiasFromRECO+AlCaPCCRandomFromRECO+SiPixelCalZeroBias+SiPixelCalSingleMuon,ALCA:PromptCalibProd+PromptCalibProdSiStrip+PromptCalibProdSiStripGains+PromptCalibProdSiStripGainsAAG+PromptCalibProdSiStripHitEff+PromptCalibProdSiPixelAli+PromptCalibProdSiPixelAliHG+PromptCalibProdSiPixel+PromptCalibProdSiPixelLA',
                  '--customise': 'Alignment/CommonAlignmentProducer/customizeLSNumberFilterForRelVals.doNotFilterLS',
                  '--conditions':'auto:run2_data',
                  '--datatier':'ALCARECO',
                  '--eventcontent':'ALCARECO',
                  '--triggerResultsProcess': 'RECO'}

steps['ALCAEXPRUN3']={'-s':'ALCAOUTPUT:@allForPrompt+@allForExpress,ALCA:PromptCalibProd+PromptCalibProdSiStrip+PromptCalibProdSiStripGains+PromptCalibProdSiStripGainsAAG+PromptCalibProdSiStripHitEff+PromptCalibProdSiPixelAli+PromptCalibProdSiPixel+PromptCalibProdSiPixelLA+PromptCalibProdBeamSpotHP+PromptCalibProdBeamSpotHPLowPU',
                  '--conditions':'auto:run3_data',
                  '--datatier':'ALCARECO',
                  '--eventcontent':'ALCARECO',
                  '--triggerResultsProcess': 'RECO'}
steps['ALCAEXPCOSMICSRUN3']={'-s':'ALCAOUTPUT:@allForExpressCosmics,ALCA:PromptCalibProdSiStrip+PromptCalibProdSiPixelLAMCS+PromptCalibProdSiStripLA',
                             '-n':1000,
                             '--scenario':'cosmics',
                             '--era':'Run3',
                             '--datatier':'ALCARECO',
                             '--eventcontent':'ALCARECO',
                             '--conditions':'auto:run3_data_express',
                             '--triggerResultsProcess': 'RECO'}

steps['ALCAEXPHI']=merge([{'-s':'ALCA:PromptCalibProd+PromptCalibProdSiStrip+PromptCalibProdSiStripGains+PromptCalibProdSiStripGainsAAG',
                  '--scenario':'HeavyIons'},steps['ALCAEXP']])
steps['ALCAEXPTE']={'-s':'ALCA:PromptCalibProdEcalPedestals',
                    '--conditions':'auto:run2_data',
                    '--datatier':'ALCARECO',
                    '--eventcontent':'ALCARECO',
                    '--triggerResultsProcess': 'RECO'}

steps['ALCARECOEXPR3']=merge([{'-s':'ALCAOUTPUT:SiPixelCalZeroBias+SiStripCalZeroBias+SiStripCalMinBias+SiStripCalMinBiasAAG+TkAlMinBias,ALCA:PromptCalibProd+PromptCalibProdSiStrip+PromptCalibProdSiPixelAli+PromptCalibProdSiPixelAliHG+PromptCalibProdSiStripGains+PromptCalibProdSiStripGainsAAG+PromptCalibProdSiPixel',
                               '--conditions':'auto:run3_data_express',
                               '--scenario':'pp',
                               '--era':'Run3',
                               '--datatier':'ALCARECO',
                               '--eventcontent':'ALCARECO',
                               '--triggerResultsProcess': 'RECO',
                               '--customise':'Configuration/DataProcessing/RecoTLR.customiseExpress,Alignment/CommonAlignmentProducer/customizeLSNumberFilterForRelVals.doNotFilterLS'},steps['RECODR3']])

steps['ALCARECOPROMPTR3']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCA:SiStripCalZeroBias+SiStripCalMinBias+TkAlMinBias+HcalCalHO+HcalCalIterativePhiSym+HcalCalHBHEMuonProducerFilter+HcalCalIsoTrkProducerFilter,DQM:@standardDQMFakeHLT',
                                  '--conditions':'auto:run3_data_prompt',
                                  '--scenario':'pp',
                                  '--era':'Run3',
                                  '--datatier':'RECO,MINIAOD,ALCARECO,DQMIO',
                                  '--eventcontent':'RECO,MINIAOD,ALCARECO,DQM',
                                  '--triggerResultsProcess': 'RECO',
                                  '--customise':'Configuration/DataProcessing/RecoTLR.customisePrompt'},steps['RECODR3']])

steps['ALCAECALPHISYM']={'--scenario':'pp',
                         '--era':'Run3',
                         '--conditions':'auto:run3_data_prompt',
                         '--datatier':'ALCARECO',
                         '--eventcontent':'ALCARECO',
                         '-n':'-1',
                         '-s':'ALCA:EcalPhiSymByRun',
                         '--customise':'Calibration/EcalCalibAlgos/EcalPhiSymRecoSequence_cff'}

steps['ALCASPLITRUN3']={'-s':'ALCAOUTPUT:@allForPrompt+@allForExpress,ALCA:PromptCalibProd+PromptCalibProdSiStrip+PromptCalibProdSiStripGains+PromptCalibProdSiStripGainsAAG+PromptCalibProdSiStripHitEff+PromptCalibProdSiPixelAli+PromptCalibProdSiPixel+PromptCalibProdSiPixelLA+PromptCalibProdBeamSpotHP+PromptCalibProdBeamSpotHPLowPU',
                  '--conditions':'auto:run3_data',
                  '--datatier':'ALCARECO',
                  '--eventcontent':'ALCARECO',
                  '--triggerResultsProcess': 'RECO'}

# step4
step4Defaults = { '-s'            : 'ALCA:TkAlMuonIsolated+TkAlMinBias+EcalCalZElectron+EcalCalWElectron+HcalCalIsoTrk+MuAlCalIsolatedMu+MuAlZMuMu+MuAlOverlaps',
                  '-n'            : 1000,
                  '--conditions'  : 'auto:run1_mc',
                  '--datatier'    : 'ALCARECO',
                  '--eventcontent': 'ALCARECO',
                  }
step4Up2015Defaults = {
                        '-s'            : 'ALCA:TkAlMuonIsolated+TkAlMinBias+MuAlOverlaps+EcalESAlign+EcalTrg',
                        '-n'            : 1000,
                        '--conditions'  : 'auto:run2_mc',
                        '--era'         : 'Run2_2016',
                        '--datatier'    : 'ALCARECO',
                        '--eventcontent': 'ALCARECO',
                  }

steps['RERECOPU']=steps['RERECOPU1']

steps['ALCATT']=merge([{'--filein':'file:step3.root','-s':'ALCA:TkAlMuonIsolated+TkAlMinBias+MuAlCalIsolatedMu+MuAlZMuMu+MuAlOverlaps'},step4Defaults])
steps['ALCATTUP15']=merge([{'--filein':'file:step3.root'},step4Up2015Defaults])
steps['ALCAMIN']=merge([{'-s':'ALCA:TkAlMinBias','--filein':'file:step3.root'},stCond,step4Defaults])
steps['ALCAMINUP15']=merge([{'-s':'ALCA:TkAlMinBias','--filein':'file:step3.root'},step4Up2015Defaults])
steps['ALCACOS']=merge([{'-s':'ALCA:TkAlCosmics0T+MuAlGlobalCosmics+HcalCalHOCosmics'},stCond,step4Defaults])
steps['ALCABH']=merge([{'-s':'ALCA:TkAlBeamHalo+MuAlBeamHaloOverlaps+MuAlBeamHalo'},stCond,step4Defaults])
steps['ALCABH_UP18']=merge([{'--conditions':'auto:phase1_2018_cosmics','-s':'ALCA:TkAlBeamHalo+MuAlBeamHaloOverlaps+MuAlBeamHalo','--era':'Run2_2018'},step4Up2015Defaults])
steps['ALCABH_UP21']=merge([{'--conditions':'auto:phase1_2022_cosmics','-s':'ALCA:TkAlBeamHalo+MuAlBeamHaloOverlaps+MuAlBeamHalo','--era':'Run3'},step4Up2015Defaults])

steps['ALCAHAL']=merge([{'-s':'ALCA:TkAlBeamHalo+MuAlBeamHaloOverlaps+MuAlBeamHalo'},step4Up2015Defaults])
steps['ALCACOS_UP15']=merge([{'--conditions':'auto:run2_mc_cosmics','-s':'ALCA:TkAlBeamHalo+MuAlBeamHaloOverlaps+MuAlBeamHalo'},step4Up2015Defaults])
steps['ALCACOS_UP16']=merge([{'--conditions':'auto:run2_mc_cosmics','-s':'ALCA:TkAlCosmics0T+SiStripCalCosmics+SiStripCalCosmicsNano+SiPixelCalCosmics+TkAlBeamHalo+MuAlBeamHaloOverlaps+MuAlBeamHalo','--era':'Run2_2016'},step4Up2015Defaults])
steps['ALCACOS_UP17']=merge([{'--conditions':'auto:phase1_2017_cosmics','-s':'ALCA:TkAlCosmics0T+SiStripCalCosmics+SiStripCalCosmicsNano+SiPixelCalCosmics+TkAlBeamHalo+MuAlBeamHaloOverlaps+MuAlBeamHalo','--era':'Run2_2017'},step4Up2015Defaults])
steps['ALCACOS_UP18']=merge([{'--conditions':'auto:phase1_2018_cosmics','-s':'ALCA:TkAlCosmics0T+SiStripCalCosmics+SiStripCalCosmicsNano+SiPixelCalCosmics+TkAlBeamHalo+MuAlBeamHaloOverlaps+MuAlBeamHalo','--era':'Run2_2018'},step4Up2015Defaults])
steps['ALCACOS_UP21']=merge([{'--conditions':'auto:phase1_2022_cosmics','-s':'ALCA:TkAlCosmics0T+SiStripCalCosmics+SiStripCalCosmicsNano+SiPixelCalCosmics+TkAlBeamHalo+MuAlBeamHaloOverlaps+MuAlBeamHalo','--era':'Run3'},step4Up2015Defaults])
steps['ALCACOS_UP21_0T']=merge([{'--magField':'0T','--conditions':'auto:phase1_2022_cosmics_0T'},steps['ALCACOS_UP21']])
steps['ALCAHARVD']={'-s':'ALCAHARVEST:BeamSpotByRun+BeamSpotByLumi+SiStripQuality',
                    '--conditions':'auto:run1_data',
                    '--scenario':'pp',
                    '--data':'',
                    '--filein':'file:PromptCalibProd.root'}

steps['ALCAHARVD1']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProd']),
                     '--conditions':'auto:run1_data',
                     '--scenario':'pp',
                     '--data':'',
                     '--filein':'file:PromptCalibProd.root'}
steps['ALCAHARVD1HI']=merge([{'--scenario':'HeavyIons'},steps['ALCAHARVD1']])

steps['ALCAHARVD2']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProdSiStrip']),
                     '--conditions':'auto:run1_data',
                     '--scenario':'pp',
                     '--data':'',
                     '--filein':'file:PromptCalibProdSiStrip.root'}

steps['ALCAHARVD2HI']=merge([{'--scenario':'HeavyIons'},steps['ALCAHARVD2']])
steps['ALCAHARVD2COS']=merge([{'--scenario':'cosmics',
                               '--conditions':'auto:run3_data_express'},steps['ALCAHARVD2']])

steps['ALCAHARVD3']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProdSiStripGains']),
                     '--conditions':'auto:run1_data',
                     '--scenario':'pp',
                     '--data':'',
                     '--filein':'file:PromptCalibProdSiStripGains.root'}

steps['ALCAHARVD3COS']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProdSiStripLA']),
                     '--conditions':'auto:run3_data_express',
                     '--scenario':'cosmics',
                     '--data':'',
                     '--filein':'file:PromptCalibProdSiStripLA.root'}

steps['ALCAHARVD3HI']=merge([{'--scenario':'HeavyIons'},steps['ALCAHARVD3']])

steps['ALCAHARVD4']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProdSiPixelAli']),
                     '--conditions':'auto:run1_data',
                     '--scenario':'pp',
                     '--data':'',
                     '--filein':'file:PromptCalibProdSiPixelAli.root'}

steps['ALCAHARVD4HI']=merge([{'--scenario':'HeavyIons'},steps['ALCAHARVD4']])

steps['ALCAHARVD5']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProdSiStripGainsAAG']),
                     '--conditions':'auto:run1_data',
                     '--scenario':'pp',
                     '--data':'',
                     '--filein':'file:PromptCalibProdSiStripGainsAAG.root'}

steps['ALCAHARVD6']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProdSiPixel']),
                     '--conditions':'auto:run1_data',
                     '--scenario':'pp',
                     '--data':'',
                     '--filein':'file:PromptCalibProdSiPixel.root'}

steps['ALCAHARVD7']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProdSiPixelLA']),
                     '--conditions':'auto:run1_data',
                     '--scenario':'pp',
                     '--data':'',
                     '--filein':'file:PromptCalibProdSiPixelLA.root'}

steps['ALCAHARVD8']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProdSiStripHitEff']),
                     '--conditions':'auto:run1_data',
                     '--scenario':'pp',
                     '--data':'',
                     '--filein':'file:PromptCalibProdSiStripHitEff.root'}

steps['ALCAHARVD5HI']=merge([{'--scenario':'HeavyIons'},steps['ALCAHARVD5']])
steps['ALCAHARVDTE']={'-s':'ALCAHARVEST:%s'%(autoPCL['PromptCalibProdEcalPedestals']),
                     '--conditions':'auto:run2_data',
                     '--scenario':'pp',
                     '--data':'',
                     '--filein':'file:PromptCalibProdEcalPedestals.root'}

steps['ALCAHARVDEXPRUN3']={'-s':'ALCAHARVEST:SiStripQuality+SiStripHitEff+SiStripGains+SiStripGainsAAG+SiPixelAli+SiPixelQuality+SiPixelLA+BeamSpotHPByRun+BeamSpotHPByLumi+BeamSpotHPLowPUByRun+BeamSpotHPLowPUByLumi',
                    '--conditions':'auto:run3_data',
                    '--scenario':'pp',
                    '--data':'',
                    '--filein':'file:PromptCalibProd.root'}


steps['RECOHISt4']=steps['RECOHI2015']

steps['ALCANZS']=merge([{'-s':'ALCA:HcalCalMinBias','--mc':''},step4Defaults])
steps['HARVESTGEN']={'-s':'HARVESTING:genHarvesting',
                     '--conditions':'auto:run2_mc',
                     '--mc':'',
                     '--era' :'Run2_2016',
                     '--filetype':'DQM',
                     '--filein':'file:step1_inDQM.root'
                  }

steps['HARVESTGEN2']=merge([{'--filein':'file:step2_inDQM.root'},steps['HARVESTGEN']])


#data
steps['HARVESTD']={'-s':'HARVESTING:@standardDQMFakeHLT+@miniAODDQM',
                   '--conditions':'auto:run1_data',
                   '--data':'',
                   '--filetype':'DQM',
                   '--scenario':'pp'}

steps['HARVESTDR1']={'-s':'HARVESTING:@standardDQMFakeHLT+@miniAODDQM',
                   '--conditions':'auto:run1_data',
                   '--data':'',
                   '--filetype':'DQM',
                   '--scenario':'pp'}

steps['HARVESTDreHLT'] = merge([ {'--conditions':'auto:run1_data_%s'%menu,'--hltProcess':'reHLT'}, steps['HARVESTD'] ])
steps['HARVESTDR1reHLT'] = merge([ {'--conditions':'auto:run1_data_%s'%menu,'--hltProcess':'reHLT'}, steps['HARVESTDR1'] ])
steps['HARVESTDR2'] = merge([ {'--conditions':'auto:run2_data_relval'}, steps['HARVESTD'] ])
steps['HARVESTDR2_skimDoubleEG'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@ecal+@egamma'}, steps['HARVESTDR2'] ])
steps['HARVESTDR2_skimJetHT'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@jetmet+@hcalOnly'}, steps['HARVESTDR2'] ])
steps['HARVESTDR2_skimMET'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@jetmet+@hcalOnly'}, steps['HARVESTDR2'] ])
steps['HARVESTDR2_skimMuonEG'] = merge([ {'-s':'HARVESTING:@commonFakeHLT'}, steps['HARVESTDR2'] ])
steps['HARVESTDR2_skimSinglePh'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@ecal+@egamma'}, steps['HARVESTDR2'] ])
steps['HARVESTDR2_skimMuOnia'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@muon'}, steps['HARVESTDR2'] ])
steps['HARVESTDR2ZB'] = merge([ {'--conditions':'auto:run2_data_relval','-s':'HARVESTING:@rerecoZeroBiasFakeHLT+@miniAODDQM'}, steps['HARVESTD'] ])

steps['HARVEST2016'] = merge([ {'--era':'Run2_2016'}, steps['HARVESTDR2'] ])
steps['HARVEST2016_L1TEgDQM'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@ecal+@egamma+@L1TEgamma'}, steps['HARVEST2016'] ])
steps['HARVEST2016_L1TMuDQM'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@muon+@lumi+@L1TMuon'}, steps['HARVEST2016'] ])
steps['HARVEST2016_skimDoubleEG'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@ecal+@egamma'}, steps['HARVEST2016'] ])
steps['HARVEST2016_skimJetHT'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@jetmet+@hcalOnly'}, steps['HARVEST2016'] ])
steps['HARVEST2016_skimMET'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@jetmet+@hcalOnly'}, steps['HARVEST2016'] ])
steps['HARVEST2016_skimMuonEG'] = merge([ {'-s':'HARVESTING:@commonFakeHLT'}, steps['HARVEST2016'] ])
steps['HARVEST2016_skimDisplacedJet'] = merge([ {'-s':'HARVESTING:@commonFakeHLT'}, steps['HARVEST2016'] ])
steps['HARVEST2016_skimSinglePh'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@ecal+@egamma'}, steps['HARVEST2016'] ])
steps['HARVEST2016_skimMuOnia'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@muon'}, steps['HARVEST2016'] ])
steps['HARVEST2016_skimCharmonium'] = merge([ {'-s':'HARVESTING:@commonFakeHLT'}, steps['HARVEST2016'] ])

steps['HARVEST2017'] = merge([ {'--conditions':'auto:run2_data_relval','--era':'Run2_2017','--conditions':'auto:run2_data'}, steps['HARVESTD'] ])
steps['HARVEST2017ZB'] = merge([ {'--conditions':'auto:run2_data_relval','--era':'Run2_2017','--conditions':'auto:run2_data','-s':' HARVESTING:@rerecoZeroBiasFakeHLT+@miniAODDQM'}, steps['HARVESTD'] ])
steps['HARVEST2017_L1TEgDQM'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@ecal+@egamma+@L1TEgamma'}, steps['HARVEST2017'] ])
steps['HARVEST2017_L1TMuDQM'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@muon+@lumi+@L1TMuon'}, steps['HARVEST2017'] ])
steps['HARVEST2017_skimDoubleEG'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@ecal+@egamma'}, steps['HARVEST2017'] ])
steps['HARVEST2017_skimJetHT'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@jetmet+@hcalOnly'}, steps['HARVEST2017'] ])
steps['HARVEST2017_skimMET'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@jetmet+@hcalOnly'}, steps['HARVEST2017'] ])
steps['HARVEST2017_skimMuonEG'] = merge([ {'-s':'HARVESTING:@commonFakeHLT'}, steps['HARVEST2017'] ])
steps['HARVEST2017_skimDisplacedJet'] = merge([ {'-s':'HARVESTING:@commonFakeHLT'}, steps['HARVEST2017'] ])
steps['HARVEST2017_skimSinglePh'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@ecal+@egamma'}, steps['HARVEST2017'] ])
steps['HARVEST2017_skimMuOnia'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@muon'}, steps['HARVEST2017'] ])
steps['HARVEST2017_skimCharmonium'] = merge([ {'-s':'HARVESTING:@commonFakeHLT'}, steps['HARVEST2017'] ])

steps['HARVEST2018'] = merge([ {'--conditions':'auto:run2_data_relval','--era':'Run2_2018','--conditions':'auto:run2_data',}, steps['HARVESTD'] ])
steps['HARVEST2018ZB'] = merge([ {'--conditions':'auto:run2_data_relval','--era':'Run2_2018','--conditions':'auto:run2_data','-s':'HARVESTING:@rerecoZeroBiasFakeHLT+@miniAODDQM'}, steps['HARVESTD'] ])
steps['HARVEST2018_Prompt'] = merge([ {'--conditions':'auto:run2_data_relval','--era':'Run2_2018','--conditions':'auto:run2_data',}, steps['HARVESTD'] ])
steps['HARVEST2018_PromptZB'] = merge([ {'--conditions':'auto:run2_data_relval','--era':'Run2_2018','--conditions':'auto:run2_data','-s':'HARVESTING:@rerecoZeroBiasFakeHLT+@miniAODDQM'}, steps['HARVESTD'] ])
steps['HARVEST2018_L1TEgDQM'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@ecal+@egamma+@L1TEgamma'}, steps['HARVEST2018'] ])
steps['HARVEST2018_L1TEgDQM_Prompt'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@ecal+@egamma+@L1TEgamma'}, steps['HARVEST2018_Prompt'] ])
steps['HARVEST2018_L1TMuDQM'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@muon+@lumi+@L1TMuon'}, steps['HARVEST2018'] ])
steps['HARVEST2018_L1TMuDQM_Prompt'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@muon+@lumi+@L1TMuon'}, steps['HARVEST2018_Prompt'] ])
steps['HARVEST2018_pixelTrackingOnly'] = merge([ {'-s':'HARVESTING:@pixelTrackingOnlyDQM'}, steps['HARVEST2018'] ])
steps['HARVEST2018_hBStar'] = merge([ {'--era' : 'Run2_2018_highBetaStar'}, steps['HARVEST2018'] ])
steps['HARVEST2018_HEfail'] = merge([ {'--conditions':'auto:run2_data_HEfail'}, steps['HARVEST2018'] ])
steps['HARVEST2018_skimJetHT'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@jetmet+@hcalOnly'}, steps['HARVEST2018'] ])
steps['HARVEST2018_skimMET'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@jetmet+@hcalOnly'}, steps['HARVEST2018'] ])
steps['HARVEST2018_skimMuonEG'] = merge([ {'-s':'HARVESTING:@commonFakeHLT'}, steps['HARVEST2018'] ])
steps['HARVEST2018_skimMuOnia'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@muon'}, steps['HARVEST2018'] ])
steps['HARVEST2018_skimDisplacedJet'] = merge([ {'-s':'HARVESTING:@commonFakeHLT'}, steps['HARVEST2018'] ])
steps['HARVEST2018_skimCharmonium'] = merge([ {'-s':'HARVESTING:@commonFakeHLT'}, steps['HARVEST2018'] ])
steps['HARVEST2018_skimParkingBPH'] = merge([ {'-s':'HARVESTING:@commonFakeHLT'}, steps['HARVEST2018'] ])
steps['HARVEST2018_HEfail_skimJetHT'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@jetmet+@hcalOnly'}, steps['HARVEST2018_HEfail'] ])
steps['HARVEST2018_skimJetHT_Prompt'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@jetmet+@hcalOnly'}, steps['HARVEST2018_Prompt'] ])
steps['HARVEST2018_skimMET_Prompt'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@jetmet+@hcalOnly'}, steps['HARVEST2018_Prompt'] ])
steps['HARVEST2018_skimMuonEG_Prompt'] = merge([ {'-s':'HARVESTING:@commonFakeHLT'}, steps['HARVEST2018_Prompt'] ])
steps['HARVEST2018_skimMuOnia_Prompt'] = merge([ {'-s':'HARVESTING:@commonFakeHLT+@muon'}, steps['HARVEST2018_Prompt'] ])
steps['HARVEST2018_skimDisplacedJet_Prompt'] = merge([ {'-s':'HARVESTING:@commonFakeHLT'}, steps['HARVEST2018_Prompt'] ])
steps['HARVEST2018_skimCharmonium_Prompt'] = merge([ {'-s':'HARVESTING:@commonFakeHLT'}, steps['HARVEST2018_Prompt'] ])
steps['HARVEST2018_skimParkingBPH_Prompt'] = merge([ {'-s':'HARVESTING:@commonFakeHLT'}, steps['HARVEST2018_Prompt'] ])

steps['HARVEST2018_L1TEgDQM_MULTIRUN'] = merge([ {
        '--customise':"Configuration/StandardSequences/DQMSaverAtJobEnd_cff",
        # hardcode the input files since we need multiple, from each of the RECO steps.
        '--filein':"file:step6_inDQM.root,file:step3_inDQM.root",
    }, steps['HARVEST2018_L1TEgDQM'] ])


#RUN3
steps['DQMHLTonAOD_2017']={
    '-s':'DQM:offlineHLTSourceOnAOD', ### DQM-only workflow on AOD input: for HLT
    '--conditions':'auto:run2_data',
    '--data':'',
    '--eventcontent':'DQM',
    '--datatier':'DQMIO',
    '--era':'Run2_2017',
    '--fileout':'DQMHLTonAOD.root',
    '--procModifiers':'tau_readOldDiscriminatorFormat',
    }
steps['DQMHLTonAODextra_2017'] = merge([ {'-s':'DQM:offlineHLTSourceOnAODextra'}, steps['DQMHLTonAOD_2017'] ])

steps['DQMHLTonRAWAOD_2017']={
    '--process':'reHLT',
    '-s':'HLT:@relval2017,DQM:offlineHLTSourceOnAOD',
    '--conditions':'auto:run2_data',
    '--data':'',
    '--eventcontent':'DQM',
    '--datatier':'DQMIO',
    '--era':'Run2_2017',
    '--secondfilein':'filelist:step1_dasparentquery.log',
    '--customise_commands':'"process.HLTAnalyzerEndpath.remove(process.hltL1TGlobalSummary)"',
    '--fileout':'DQMHLTonAOD.root',
    '--procModifiers':'tau_readOldDiscriminatorFormat',
    }

steps['HARVESTDQMHLTonAOD_2017'] = merge([ {'--filein':'file:DQMHLTonAOD.root','-s':'HARVESTING:hltOfflineDQMClient'}, steps['HARVEST2017'] ]) ### Harvesting step for the DQM-only workflow

steps['HARVESTDDQM']=merge([{'-s':'HARVESTING:@allForPrompt'},steps['HARVESTD']])

steps['HARVESTDfst2']=merge([{'--filein':'file:step2_inDQM.root'},steps['HARVESTDR1']])

steps['HARVESTDC']={'-s':'HARVESTING:dqmHarvestingFakeHLT',
                   '--conditions':'auto:run1_data',
                   '--filetype':'DQM',
                   '--data':'',
                   '--filein':'file:step2_inDQM.root',
                   '--scenario':'cosmics'}

steps['HARVESTDCRUN2']=merge([{'--conditions':'auto:run2_data','--era':'Run2_2016'},steps['HARVESTDC']])

steps['HARVESTDR3']  = merge([{'--conditions':'auto:run3_data','--era':'Run3'}, steps['HARVESTD']])
steps['HARVESTDR3_reHLT_2022']  = merge([{'--conditions':'auto:run3_data_relval','--hltProcess':'reHLT','-s':'HARVESTING:@standardDQMFakeHLT+@miniAODDQM'}, steps['HARVESTDR3']])
steps['HARVESTDR3_reHLT_2023']  = merge([{'--conditions':'auto:run3_data_prompt_relval'}, steps['HARVESTDR3_reHLT_2022']])
steps['HARVESTD2021MB_reHLT'] = merge([{'-s':'HARVESTING:@commonSiStripZeroBias+@miniAODDQM'}, steps['HARVESTDR3_reHLT_2022'] ])
steps['HARVESTD2021ZB_reHLT'] = merge([{'-s':'HARVESTING:@rerecoZeroBias+@miniAODDQM'}, steps['HARVESTDR3_reHLT_2022'] ])
steps['HARVESTD2021HLTPhy_reHLT'] = merge([{'-s':'HARVESTING:@commonReduced+@miniAODDQM'}, steps['HARVESTDR3_reHLT_2022'] ])

steps['HARVESTDPROMPTR3']=merge([{'--conditions':'auto:run3_data_prompt','-s':'HARVESTING:dqmHarvestingFakeHLT'},steps['HARVESTDR3']])
steps['HARVESTDEXPR3']=merge([{'--conditions':'auto:run3_data_express','--filein':'file:step2_inDQM.root'},steps['HARVESTDR3']])

steps['HARVESTDCR3'] = merge([{'--conditions':'auto:run3_data','--era':'Run3'}, steps['HARVESTDC']])
steps['HARVESTDCPROMPTRUN3']=merge([{'--conditions':'auto:run3_data_prompt','--era':'Run3'},steps['HARVESTDC']])
steps['HARVESTDCEXPRUN3']=merge([{'--conditions':'auto:run3_data_express','--era':'Run3'},steps['HARVESTDC']])

steps['HARVESTDR2_REMINIAOD_data2016']=merge([{'--data':'', '-s':'HARVESTING:@miniAODDQM','--era':'Run2_2016,run2_miniAOD_80XLegacy'},steps['HARVESTDR2']])
steps['HARVESTDR2_REMININANO_data2016']=merge([{'--data':'', '-s':'HARVESTING:@miniAODDQM+@nanoAODDQM','--era':'Run2_2016'},steps['HARVESTDR2']])
steps['HARVESTDR2_REMINIAOD_data2016_HIPM']=merge([{'--era':'Run2_2016_HIPM,run2_miniAOD_80XLegacy'},steps['HARVESTDR2_REMINIAOD_data2016']])
steps['HARVESTDR2_REMINIAOD_data2016UL']=merge([{'--era':'Run2_2016', '--procModifiers':'run2_miniAOD_UL_preSummer20'},steps['HARVESTDR2_REMINIAOD_data2016']])
steps['HARVESTDR2_REMININANO_data2016UL']=merge([{'--era':'Run2_2016', '--procModifiers':'run2_miniAOD_UL_preSummer20'},steps['HARVESTDR2_REMININANO_data2016']])
steps['HARVESTDR2_REMINIAOD_data2016UL_HIPM']=merge([{'--era':'Run2_2016_HIPM'},steps['HARVESTDR2_REMINIAOD_data2016UL']])
steps['HARVESTDR2_REMININANO_data2016UL_HIPM']=merge([{'--era':'Run2_2016_HIPM'},steps['HARVESTDR2_REMININANO_data2016UL']])
steps['HARVESTDR2_REMININANO_data2016UL_HIPM_met']=merge([{'--era':'Run2_2016_HIPM'},steps['HARVESTDR2_REMININANO_data2016UL']])

steps['HARVEST2017_REMINIAOD_data2017']=merge([{'--era':'Run2_2017,run2_miniAOD_94XFall17'},steps['HARVESTDR2_REMINIAOD_data2016']])
steps['HARVEST2017_REMINIAOD_data2017UL']=merge([{'--era':'Run2_2017'},steps['HARVESTDR2_REMINIAOD_data2016UL']])
steps['HARVEST2018_REMINIAOD_data2018UL']=merge([{'--era':'Run2_2018'},steps['HARVESTDR2_REMINIAOD_data2016UL']])

steps['HARVESTDHI']={'-s':'HARVESTING:dqmHarvestingFakeHLT',
                   '--conditions':'auto:run1_data',
                   '--filetype':'DQM',
                   '--data':'',
                   '--scenario':'HeavyIons'}

steps['HARVESTDHI15']=merge([{ '--conditions':'auto:run2_data',
                               '--era':'Run2_HI',
                               '-n':-1},steps['HARVESTDHI']])

steps['HARVESTDHI18']=merge([{ '--era':'Run2_2018_pp_on_AA',
                               '--scenario':'pp' },steps['HARVESTDHI15']])

steps['HARVESTHI18MINIAOD']=merge([{'-s':'HARVESTING:@miniAODDQM',
                                    '--filein':'file:step2_inDQM.root',
                                    '--data':'',
                                    '--era' : 'Run2_2018_pp_on_AA',
                                    '--filetype':'DQM',
                                    '-n':100},hiDefaults2018_ppReco])

steps['HARVESTDHI22']=merge([{ '--era':'Run3_pp_on_PbPb',
                               '--scenario':'pp',
                               '--conditions':'auto:run3_data'},steps['HARVESTDHI15']])
                                    
steps['HARVESTDRUN3']={'-s':'HARVESTING:@standardDQMFakeHLT+@miniAODDQM+@nanoAODDQM',
                   '--conditions':'auto:run3_data',
                   '--data':'',
                   '--filetype':'DQM',
                   '--scenario':'pp'}

steps['HARVESTPROMPTRUN3']=merge([{'--data':'', '-s':'HARVESTING:@allForPrompt','--era':'Run3'},steps['HARVESTDRUN3']])


steps['HARVESTRUN3_2022']=merge([{'--data':'', '-s':'HARVESTING:@standardDQMFakeHLT+@miniAODDQM+@nanoAODDQM','--era':'Run3'},steps['HARVESTDRUN3']])
steps['HARVESTRUN3_ZB_2022']=merge([{'--data':'', '-s':'HARVESTING:@rerecoZeroBiasFakeHLT+@miniAODDQM+@nanoAODDQM','--era':'Run3'},steps['HARVESTDRUN3']])
steps['HARVESTRUN3_COS_2022']=merge([{'--data':'', '--scenario':'cosmics', '--era':'Run3', '-s':'HARVESTING:dqmHarvesting'},steps['HARVESTDRUN3']])
steps['HARVESTRUN3_2023']=merge([{'--era':'Run3_2023', '-s':'HARVESTING:@standardDQM+@miniAODDQM+@nanoAODDQM'},steps['HARVESTRUN3_2022']])
steps['HARVESTRUN3_2023B']=merge([{'--era':'Run3', '-s':'HARVESTING:@standardDQM+@miniAODDQM+@nanoAODDQM'},steps['HARVESTRUN3_2022']])
steps['HARVESTRUN3_ZB_2023B']=merge([{'--era':'Run3', '-s':'HARVESTING:@rerecoZeroBias+@miniAODDQM+@nanoAODDQM'},steps['HARVESTRUN3_2022']])
steps['HARVESTRUN3_ZB_2023']=merge([{'--era':'Run3_2023', '-s':'HARVESTING:@rerecoZeroBias+@miniAODDQM+@nanoAODDQM'},steps['HARVESTRUN3_2022']])
steps['HARVESTRUN3_COS_2023']=merge([{'--scenario':'cosmics', '--era':'Run3_2023', '-s':'HARVESTING:dqmHarvesting'},steps['HARVESTRUN3_2022']])
steps['HARVESTRUN3_pixelTrackingOnly'] = merge([ {'-s':'HARVESTING:@pixelTrackingOnlyDQM'}, steps['HARVESTRUN3_2023']])
steps['HARVESTRUN3_pixelTrackingOnlyGPUValidation'] = merge([ {'--procModifiers':'gpuValidation'}, steps['HARVESTRUN3_pixelTrackingOnly']])
steps['HARVESTRUN3_ECALOnly'] = merge([ {'-s':'HARVESTING:@ecal'}, steps['HARVESTRUN3_2023']])
steps['HARVESTRUN3_ECALOnlyGPUValidation'] = merge([ {'--procModifiers':'gpuValidation'}, steps['HARVESTRUN3_ECALOnly']])
steps['HARVESTRUN3_HCALOnly'] = merge([ {'-s':'HARVESTING:@hcalOnly'}, steps['HARVESTRUN3_2023']])
steps['HARVESTRUN3_HCALOnlyGPUValidation'] = merge([ {'--procModifiers':'gpuValidation'}, steps['HARVESTRUN3_HCALOnly']])


#MC
steps['HARVEST']={'-s':'HARVESTING:validationHarvestingNoHLT+dqmHarvestingFakeHLT',
                   '--conditions':'auto:run1_mc',
                   '--mc':'',
                   '--filetype':'DQM',
                   '--scenario':'pp'}

steps['HARVESTCOS']={'-s':'HARVESTING:dqmHarvestingFakeHLT',
                     '--conditions':'auto:run1_mc',
                     '--mc':'',
                     '--filein':'file:step3_inDQM.root',
                     '--filetype':'DQM',
                     '--scenario':'cosmics'}
steps['HARVESTHAL']={'-s'          :'HARVESTING:dqmHarvestingFakeHLT',
                     '--conditions':'auto:run2_mc',
                     '--mc'        :'',
                     '--filein'    :'file:step3_inDQM.root',
                     '--scenario'    :'cosmics',
                     '--filein':'file:step3_inDQM.root', # unnnecessary
                     '--filetype':'DQM',
                     '--era' : 'Run2_2016',
                     }
steps['HARVESTCOS_UP15']={'-s'          :'HARVESTING:dqmHarvestingFakeHLT',
                          '--conditions':'auto:run2_mc_cosmics',
                          '--mc'        :'',
                          '--filein'    :'file:step3_inDQM.root',
                          '--scenario'    :'cosmics',
                          '--filein':'file:step3_inDQM.root', # unnnecessary
                          '--filetype':'DQM',
                          '--era' : 'Run2_2016',
                          }

steps['HARVESTCOS_UP16']={'-s'          :'HARVESTING:dqmHarvestingFakeHLT',
                          '--conditions':'auto:run2_mc_cosmics',
                          '--mc'        :'',
                          '--filein'    :'file:step3_inDQM.root',
                          '--scenario'    :'cosmics',
                          '--filein':'file:step3_inDQM.root', # unnnecessary
                          '--filetype':'DQM',
                          '--era' : 'Run2_2016'
                          }

steps['HARVESTCOS_UP17']={'-s'          :'HARVESTING:dqmHarvestingFakeHLT',
                          '--conditions':'auto:phase1_2017_cosmics',
                          '--mc'        :'',
                          '--filein'    :'file:step3_inDQM.root',
                          '--scenario'    :'cosmics',
                          '--filein':'file:step3_inDQM.root', # unnnecessary
                          '--filetype':'DQM',
                          '--era' : 'Run2_2017'
                          }

steps['HARVESTCOS_UP18']={'-s'          :'HARVESTING:dqmHarvestingFakeHLT',
                          '--conditions':'auto:phase1_2018_cosmics',
                          '--mc'        :'',
                          '--filein'    :'file:step3_inDQM.root',
                          '--scenario'    :'cosmics',
                          '--filein':'file:step3_inDQM.root', # unnnecessary
                          '--filetype':'DQM',
                          '--era' : 'Run2_2018'
                          }

steps['HARVESTCOS_UP21']={'-s'          :'HARVESTING:dqmHarvesting',
                          '--conditions':'auto:phase1_2022_cosmics',
                          '--mc'        :'',
                          '--filein'    :'file:step3_inDQM.root',
                          '--scenario'    :'cosmics',
                          '--filein':'file:step3_inDQM.root', # unnnecessary
                          '--filetype':'DQM',
                          '--era' : 'Run3'
                          }

steps['HARVESTCOS_UP21_0T']=merge([{'--magField':'0T','--conditions':'auto:phase1_2022_cosmics_0T'},steps['HARVESTCOS_UP21']])

steps['HARVESTFS']={'-s':'HARVESTING:validationHarvestingFS',
                   '--conditions':'auto:run1_mc',
                   '--mc':'',
                    '--fast':'',
                    '--filetype':'DQM',
                   '--scenario':'pp'}

steps['HARVESTHI2023PPRECO']=merge([hiDefaults2023_ppReco,{'-s':'HARVESTING:validationHarvestingNoHLT+dqmHarvestingFakeHLT',
                                                           '--filein':'file:step3_inDQM.root',
                                                           '--mc':'',
                                                           '--era' : 'Run3_pp_on_PbPb_2023',
                                                           '--filetype':'DQM'}])

steps['HARVESTHI2023PPRECO4']=merge([hiDefaults2023_ppReco,{'-s':'HARVESTING:validationHarvestingNoHLT+dqmHarvestingFakeHLT',
                                                            '--filein':'file:step4_inDQM.root',
                                                            '--mc':'',
                                                            '--era' : 'Run3_pp_on_PbPb_2023',
                                                            '--filetype':'DQM'}])
steps['HARVESTHI2023PPRECOMINIAOD']=merge([{'-s':'HARVESTING:@miniAODValidation+@miniAODDQM',
                                                                  '--filein':'file:step2_inDQM.root',
                                                                  '--mc':'',
                                                                  '--era' : 'Run3_pp_on_PbPb_2023',
                                                                  '--filetype':'DQM',
                                                                  '-n':100},hiDefaults2023_ppReco])

steps['HARVESTHI2022PPRECO']=merge([hiDefaults2022_ppReco,{'-s':'HARVESTING:validationHarvestingNoHLT+dqmHarvestingFakeHLT',
                                                           '--filein':'file:step3_inDQM.root',
                                                           '--mc':'',
                                                           '--era' : 'Run3_pp_on_PbPb',
                                                           '--filetype':'DQM'}])

steps['HARVESTHI2022PPRECO4']=merge([hiDefaults2022_ppReco,{'-s':'HARVESTING:validationHarvestingNoHLT+dqmHarvestingFakeHLT',
                                                            '--filein':'file:step4_inDQM.root',
                                                            '--mc':'',
                                                            '--era' : 'Run3_pp_on_PbPb',
                                                            '--filetype':'DQM'}])
steps['HARVESTHI2022PPRECOMINIAOD']=merge([{'-s':'HARVESTING:@miniAODValidation+@miniAODDQM',
                                                                  '--filein':'file:step2_inDQM.root',
                                                                  '--mc':'',
                                                                  '--era' : 'Run3_pp_on_PbPb',
                                                                  '--filetype':'DQM',
                                                                  '-n':100},hiDefaults2022_ppReco])

steps['HARVESTHI2018PPRECO']=merge([hiDefaults2018_ppReco,{'-s':'HARVESTING:validationHarvestingNoHLT+dqmHarvestingFakeHLT',
                                                           '--filein':'file:step3_inDQM.root',
                                                           '--mc':'',
                                                           '--era' : 'Run2_2018_pp_on_AA',
                                                           '--filetype':'DQM'}])

steps['HARVESTHI2018PPRECOMINIAOD']=merge([{'-s':'HARVESTING:@miniAODValidation+@miniAODDQM',
                                                                  '--filein':'file:step2_inDQM.root',
                                                                  '--mc':'',
                                                                  '--era' : 'Run2_2018_pp_on_AA',
                                                                  '--filetype':'DQM',
                                                                  '-n':100},hiDefaults2018_ppReco])

steps['HARVESTHI2018']=merge([hiDefaults2018,{'-s':'HARVESTING:validationHarvesting+dqmHarvestingFakeHLT',
                    '--mc':'',
                    '--era' : 'Run2_2017',
                    '--filetype':'DQM',
                    '--scenario':'HeavyIons'}])
steps['HARVESTHI2017']=merge([hiDefaults2017,{'-s':'HARVESTING:validationHarvestingNoHLT+dqmHarvestingFakeHLT',
                    '--mc':'',
                    '--era' : 'Run2_2017_pp_on_XeXe',
                    '--filetype':'DQM'}])
steps['HARVESTHI2015']=merge([hiDefaults2015,{'-s':'HARVESTING:validationHarvesting+dqmHarvestingFakeHLT',
                    '--mc':'',
                    '--era' : 'Run2_2016,Run2_HI',
                    '--filetype':'DQM',
                    '--scenario':'HeavyIons'}])
steps['HARVESTHI2011']=merge([hiDefaults2011,{'-s':'HARVESTING:validationHarvesting+dqmHarvestingFakeHLT',
                                              '--mc':'',
                                              '--filetype':'DQM'}])

steps['HARVESTPPREF2017']=merge([ppRefDefaults2017,{'-s':'HARVESTING:validationHarvestingNoHLT+dqmHarvestingFakeHLT',
                    '--mc':'',
                    '--era' : 'Run2_2017_ppRef',
                    '--filetype':'DQM'}])

steps['HARVESTUP15']={
    # '-s':'HARVESTING:validationHarvesting+dqmHarvesting', # todo: remove UP from label
    '-s':'HARVESTING:@standardValidationNoHLT+@standardDQMFakeHLT+@miniAODValidation+@miniAODDQM', # todo: remove UP from label
    '--conditions':'auto:run2_mc',
    '--mc':'',
    '--era' : 'Run2_2016',
    '--filetype':'DQM',
    }
steps['HARVESTUP15_L1TEgDQM']=merge([{'-s':'HARVESTING:@standardValidationNoHLT+@standardDQMFakeHLT+@miniAODValidation+@miniAODDQM+@L1TEgamma'},steps['HARVESTUP15']])
steps['HARVESTUP15_L1TMuDQM']=merge([{'-s':'HARVESTING:@standardValidationNoHLT+@standardDQMFakeHLT+@miniAODValidation+@miniAODDQM+@L1TMuon'},steps['HARVESTUP15']])

steps['HARVESTMINUP15']=merge([{'-s':'HARVESTING:validationHarvestingNoHLT+dqmHarvestingFakeHLT'},steps['HARVESTUP15']])

steps['HARVESTUP15_PU25']=steps['HARVESTUP15']
steps['HARVESTUP15_PU25_L1TEgDQM']=steps['HARVESTUP15_L1TEgDQM']
steps['HARVESTUP15_PU25_L1TMuDQM']=steps['HARVESTUP15_L1TMuDQM']


steps['HARVESTUP15_PU50']=merge([{'-s':'HARVESTING:@standardValidationNoHLT+@standardDQMFakeHLT+@miniAODValidation+@miniAODDQM','--era' : 'Run2_50ns'},steps['HARVESTUP15']])
steps['HARVESTUP15_PU50_L1TEgDQM']=merge([{'-s':'HARVESTING:@standardValidationNoHLT+@standardDQMFakeHLT+@miniAODValidation+@miniAODDQM+@L1TEgamma'},steps['HARVESTUP15_PU50']])
steps['HARVESTUP15_PU50_L1TMuDQM']=merge([{'-s':'HARVESTING:@standardValidationNoHLT+@standardDQMFakeHLT+@miniAODValidation+@miniAODDQM+@L1TMuon'},steps['HARVESTUP15_PU50']])

steps['HARVESTUP15_trackingOnly']=merge([{'-s': 'HARVESTING:@trackingOnlyValidation+@trackingOnlyDQM'}, steps['HARVESTUP15']])

steps['HARVESTUP17']=merge([{'--conditions':'auto:phase1_2017_realistic','--era' : 'Run2_2017','--geometry' : 'DB:Extended'},steps['HARVESTUP15']])
steps['HARVESTUP18']=merge([{'--conditions':'auto:phase1_2018_realistic','--era' : 'Run2_2018','--geometry' : 'DB:Extended'},steps['HARVESTUP15']])
steps['HARVESTUP18_L1TEgDQM']=merge([{'-s':'HARVESTING:@standardValidationNoHLT+@standardDQMFakeHLT+@miniAODValidation+@miniAODDQM+@L1TEgamma'},steps['HARVESTUP18']])
steps['HARVESTUP18_L1TMuDQM']=merge([{'-s':'HARVESTING:@standardValidationNoHLT+@standardDQMFakeHLT+@miniAODValidation+@miniAODDQM+@L1TMuon'},steps['HARVESTUP18']])

steps['HARVESTUP17_PU25']=steps['HARVESTUP17']
steps['HARVESTUP18_PU25']=steps['HARVESTUP18']
steps['HARVESTUP18_PU25_L1TEgDQM']=steps['HARVESTUP18_L1TEgDQM']
steps['HARVESTUP18_PU25_L1TMuDQM']=steps['HARVESTUP18_L1TMuDQM']

#Run-Dependent harvesting; using MultiRun harvesting + force RunNumber=1 for GUI MC
step_harvest_MRH_RD = {'--harvesting':'AtJobEnd', '--procModifiers':'runDependent'}
steps['HARVESTUP18_PU25_RD']=merge([step_harvest_MRH_RD,steps['HARVESTUP18']])
steps['HARVESTUP18_PU25_L1TEgDQM_RD']=merge([step_harvest_MRH_RD,steps['HARVESTUP18_L1TEgDQM']])
steps['HARVESTUP18_PU25_L1TMuDQM_RD']=merge([step_harvest_MRH_RD,steps['HARVESTUP18_L1TMuDQM']])

steps['HARVESTDR2_REMINIAOD_mc2016']=merge([{'-s':'HARVESTING:@miniAODValidation+@miniAODDQM','--era':'Run2_2016,run2_miniAOD_80XLegacy'},steps['HARVESTUP15']])
steps['HARVESTUP17_REMINIAOD_mc2017']=merge([{'-s':'HARVESTING:@miniAODValidation+@miniAODDQM','--era':'Run2_2017,run2_miniAOD_94XFall17'},steps['HARVESTUP17']])
# UL setup uses proc modifiers
steps['HARVESTDR2_REMINIAOD_mc2016UL_preVFP']=merge([{'--conditions':'auto:run2_mc_pre_vfp','--era':'Run2_2016_HIPM','--procModifiers':'run2_miniAOD_UL_preSummer20'},
                                                     steps['HARVESTDR2_REMINIAOD_mc2016']])
steps['HARVESTDR2_REMINIAOD_mc2016UL_postVFP']=merge([{'--conditions':'auto:run2_mc','--era':'Run2_2016'},steps['HARVESTDR2_REMINIAOD_mc2016UL_preVFP']])
steps['HARVESTUP17_REMINIAOD_mc2017UL']=merge([{'--era':'Run2_2017','--procModifiers':'run2_miniAOD_UL_preSummer20'},steps['HARVESTUP17_REMINIAOD_mc2017']])
steps['HARVESTUP18_REMINIAOD_mc2018UL']=merge([{'--conditions':'auto:phase1_2018_realistic','--era':'Run2_2018'},steps['HARVESTUP17_REMINIAOD_mc2017UL']])

# for Run1 PPb data workflow
steps['HARVEST_PPbData']=merge([{'--conditions':'auto:run1_data','-s':'HARVESTING:dqmHarvestingFakeHLT','--scenario':'pp','--era':'Run1_pA' }, steps['HARVESTDHI']])

# for Run2 PPb MC workflow
steps['HARVESTUP15_PPb']=merge([{'--conditions':'auto:run2_mc_pa','--era':'Run2_2016_pA'}, steps['HARVESTMINUP15']])

# unSchHarvestOverrides={'-s':'HARVESTING:@standardValidation+@standardDQM+@ExtraHLT+@miniAODValidation+@miniAODDQM'}
# steps['HARVESTmAODUP15']=merge([unSchHarvestOverrides,steps['HARVESTUP15']])

steps['HARVESTUP15FS']={'-s':'HARVESTING:validationHarvesting',
                        '--conditions':'auto:run2_mc',
                        '--fast':'',
                        '--mc':'',
                        '--era':'Run2_2016',
                        '--filetype':'DQM',
                        '--scenario':'pp'}
steps['HARVESTUP15FS_trackingOnly']=merge([{'-s': 'HARVESTING:@trackingOnlyValidation+@trackingOnlyDQM'}, steps['HARVESTUP15FS']])
steps['HARVESTUP17FS']=merge([{'--conditions':'auto:phase1_2017_realistic','--era' : 'Run2_2017_FastSim'},steps['HARVESTUP15FS']])
steps['HARVESTUP17FS_trackingOnly']=merge([{'-s': 'HARVESTING:@trackingOnlyValidation+@trackingOnlyDQM'}, steps['HARVESTUP17FS']])
steps['HARVESTUP18FS']=merge([{'--conditions':'auto:phase1_2018_realistic','--era' : 'Run2_2018_FastSim'},steps['HARVESTUP15FS']])
steps['HARVESTUP18FS_trackingOnly']=merge([{'-s': 'HARVESTING:@trackingOnlyValidation+@trackingOnlyDQM'}, steps['HARVESTUP18FS']])

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

steps['SKIMDreHLT'] = merge([ {'--conditions':'auto:run1_data_%s'%menu, '--hltProcess':'reHLT', '--filein':'file:step3.root'}, steps['SKIMD'] ])

steps['SKIMCOSD']={'-s':'SKIM:all',
                   '--conditions':'auto:run1_data',
                   '--data':'',
                   '--scenario':'cosmics',
                   '--filein':'file:step2.root',
                   '--secondfilein':'filelist:step1_dasquery.log'}

steps['RECOFROMRECO']=merge([{'-s':'RECO',
                              '--filtername':'RECOfromRECO',
                              '--process':'reRECO',
                              '--datatier':'AODSIM',
                              '--eventcontent':'AODSIM',
                              '--procModifiers':'recoFromReco',
                              },
                             stCond,step3Defaults])


steps['RECOFROMRECOSt2']=steps['RECOFROMRECO']

steps['COPYPASTE']={'-s':'NONE',
                    '--conditions':'auto:run1_mc',
                    '--output':'\'[{"t":"RAW","e":"ALL"}]\'',
                    '--customise_commands':'"process.ALLRAWoutput.fastCloning=cms.untracked.bool(False)"'}
steps['COPYPASTER3'] = merge([{'--era' : 'Run3'}, steps['COPYPASTE']])

#miniaod
stepMiniAODDefaults = { '-s'              : 'PAT',
                        '--era'           : 'Run2_2016',
                        '-n'              : '100'
                        }
stepMiniAODDataUP15 = merge([{'--conditions'   : 'auto:run1_data',
                          '--data'         : '',
                          '--datatier'     : 'MINIAOD',
                          '--eventcontent' : 'MINIAOD',
                          '--filein'       :'file:step3.root'
                          },stepMiniAODDefaults])

steps['REMINIAOD_data2016'] = merge([{'-s' : 'PAT,DQM:@miniAODDQM',
                                      '--process' : 'PAT',
                                      '--era' : 'Run2_2016,run2_miniAOD_80XLegacy',
                                      '--conditions' : 'auto:run2_data',
                                      '--data' : '',
                                      '--scenario' : 'pp',
                                      '--eventcontent' : 'MINIAOD,DQM',
                                      '--datatier' : 'MINIAOD,DQMIO'
                                      },stepMiniAODDefaults])

steps['REMININANO_data2016'] = merge([{'-s' : 'PAT,NANO,DQM:@miniAODDQM+@nanoAODDQM',
                                       '--process' : 'PATnano',
                                       '--era' : 'Run2_2016',
                                       '--conditions' : 'auto:run2_data',
                                       '--data' : '',
                                       '--scenario' : 'pp',
                                       '--eventcontent' : 'MINIAOD,NANOEDMAOD,DQM',
                                       '--datatier' : 'MINIAOD,NANOAOD,DQMIO'
                                       }])

steps['REMINIAOD_data2016_HIPM'] = merge([{'--era' : 'Run2_2016_HIPM,run2_miniAOD_80XLegacy'},steps['REMINIAOD_data2016']])
steps['REMINIAOD_data2016UL'] = merge([{'--era' : 'Run2_2016', '--procModifiers' : 'run2_miniAOD_UL_preSummer20'},steps['REMINIAOD_data2016']])
steps['REMININANO_data2016UL'] = merge([{'--era' : 'Run2_2016', '--procModifiers' : 'run2_miniAOD_UL_preSummer20'},steps['REMININANO_data2016']])
steps['REMINIAOD_data2016UL_HIPM'] = merge([{'--era' : 'Run2_2016_HIPM'},steps['REMINIAOD_data2016UL']])
steps['REMININANO_data2016UL_HIPM'] = merge([{'--era' : 'Run2_2016_HIPM'},steps['REMININANO_data2016UL']])
steps['REMININANO_data2016UL_HIPM_met'] = merge([{'--era' : 'Run2_2016_HIPM'},steps['REMININANO_data2016UL']])

stepReMiniAODData17 = merge([{'--era' : 'Run2_2017,run2_miniAOD_94XFall17'},steps['REMINIAOD_data2016']])
steps['REMINIAOD_data2017'] = stepReMiniAODData17
steps['REMINIAOD_data2017UL'] = merge([{'--era' : 'Run2_2017'},steps['REMINIAOD_data2016UL']])
steps['REMINIAOD_data2018UL'] = merge([{'--era' : 'Run2_2018'},steps['REMINIAOD_data2016UL']])

# Not sure whether the customisations are in the dict as "--customise" or "--era" so try to
# remove both. Currently premixing uses "--customise" and everything else uses "--era".
try : stepMiniAODData = remove(stepMiniAODDataUP15,'--era')
except : stepMiniAODData = remove(stepMiniAODDataUP15,'--customise')

stepMiniAODMC = merge([{'--conditions'   : 'auto:run2_mc',
                        '--mc'           : '',
                        '--era'          : 'Run2_2016',
                        '--datatier'     : 'MINIAODSIM',
                        '--eventcontent' : 'MINIAODSIM',
                        '--filein'       :'file:step3.root'
                        },stepMiniAODDefaults])

steps['REMINIAOD_mc2016'] = merge([{'-s' : 'PAT,VALIDATION:@miniAODValidation,DQM:@miniAODDQM',
                                    '--process' : 'PAT',
                                    '--era' :  'Run2_2016,run2_miniAOD_80XLegacy',
                                    '--conditions' : 'auto:run2_mc',
                                    '--mc' : '',
                                    '--scenario' : 'pp',
                                    '--eventcontent' : 'MINIAODSIM,DQM',
                                    '--datatier' : 'MINIAODSIM,DQMIO'
                                    },stepMiniAODDefaults])

steps['REMINIAOD_mc2017'] =merge([{'--conditions':'auto:phase1_2017_realistic','--era':'Run2_2017,run2_miniAOD_94XFall17'},steps['REMINIAOD_mc2016']])
# UL uses procModifiers:
steps['REMINIAOD_mc2016UL_preVFP']=merge([{'--conditions':'auto:run2_mc_pre_vfp','--era':'Run2_2016_HIPM','--procModifiers':'run2_miniAOD_UL_preSummer20'},steps['REMINIAOD_mc2016']])
steps['REMINIAOD_mc2016UL_postVFP']=merge([{'--conditions':'auto:run2_mc','--era':'Run2_2016'},steps['REMINIAOD_mc2016UL_preVFP']])
steps['REMINIAOD_mc2017UL']=merge([{'--era':'Run2_2017','--procModifiers':'run2_miniAOD_UL_preSummer20'},steps['REMINIAOD_mc2017']])
steps['REMINIAOD_mc2018UL']=merge([{'--conditions':'auto:phase1_2018_realistic','--era':'Run2_2018','--procModifiers':'run2_miniAOD_UL_preSummer20'},steps['REMINIAOD_mc2017']])

#steps['MINIAODDATA']       =merge([stepMiniAODData])
#steps['MINIAODDreHLT']     =merge([{'--conditions':'auto:run1_data_%s'%menu},stepMiniAODData])
#steps['MINIAODDATAs2']     =merge([{'--filein':'file:step2.root'},stepMiniAODData])

#MiniAOD 2016
steps['MINIAODMCUP15']     =merge([stepMiniAODMC])
#steps['MINIAODMCUP1550']   =merge([{'--conditions':'auto:run2_mc_50ns','--era':'Run2_50ns'},stepMiniAODMC])
#steps['MINIAODMCUP15HI']   =merge([{'--conditions':'auto:run2_mc_hi','--era':'Run2_HI'},stepMiniAODMC])
steps['MINIAODMCUP15FS']   =merge([{'--filein':'file:step1.root','--fast':''},stepMiniAODMC])
steps['MINIAODMCUP15FS50'] =merge([{'--conditions':'auto:run2_mc_50ns','--era':'Run2_50ns'},steps['MINIAODMCUP15FS']])
steps['DBLMINIAODMCUP15NODQM'] = merge([{'--conditions':'auto:run2_mc',
                                         '-s':'PAT',
                                         '--datatier' : 'MINIAODSIM',
                                         '--eventcontent':'MINIAOD',},stepMiniAODMC])

#MiniAOD 2017
steps['MINIAODMCUP17FS']   =merge([{'--filein':'file:step1.root','--fast':'','--conditions':'auto:phase1_2017_realistic','--era':'Run2_2017_FastSim'},stepMiniAODMC])

#MiniAOD 2018
steps['MINIAODMCUP18']   =merge([{'--conditions':'auto:phase1_2018_realistic','--era':'Run2_2018'},stepMiniAODMC])
steps['MINIAODMCUP18bParking']   =merge([{'--conditions':'auto:phase1_2018_realistic','--era':'Run2_2018,bParking'},stepMiniAODMC])
steps['MINIAODMCUP18FS']   =merge([{'--filein':'file:step1.root','--fast':'','--conditions':'auto:phase1_2018_realistic','--era':'Run2_2018_FastSim'},stepMiniAODMC])

stepNanoAODDefaults = { '-s': 'NANO,DQM:@nanoAODDQM', '-n': 1000 }
stepNanoGenDefaults = { '-s': 'NANOGEN,DQM:@nanogenDQM', '-n': 1000 }
stepNanoAODData = merge([{ '--data':'', '--eventcontent' : 'NANOAOD,DQM' ,'--datatier': 'NANOAOD,DQMIO'    }, stepNanoAODDefaults ])
stepNanoAODMC   = merge([{ '--mc':''  , '--eventcontent' : 'NANOAODSIM,DQM','--datatier': 'NANOAODSIM,DQMIO' }, stepNanoAODDefaults ])
stepNanoEDMData = merge([{ '--data':'', '--eventcontent' : 'NANOEDMAOD,DQM' ,'--datatier': 'NANOAOD,DQMIO'     }, stepNanoAODDefaults ])
stepNanoEDMMC   = merge([{ '--mc':''  , '--eventcontent' : 'NANOEDMAODSIM,DQM','--datatier': 'NANOAODSIM,DQMIO'    }, stepNanoAODDefaults ])
stepNanoEDMMCProd   = merge([{ '--mc':'', '-s': 'NANO', '--eventcontent' : 'NANOEDMAODSIM','--datatier': 'NANOAODSIM'    }, stepNanoAODDefaults ])
stepNanoGen     = merge([{ '--mc':''  , '--eventcontent' : 'NANOAODGEN,DQM','--datatier': 'NANOAODSIM,DQMIO' }, stepNanoGenDefaults ])

steps['NANOGENFromGen']   = merge([{'--conditions': 'auto:run2_mc', '--customise' : 'PhysicsTools/NanoAOD/nanogen_cff.customizeNanoGEN'}, stepNanoGen ])
steps['NANOGENFromMini']  = merge([{'--conditions': 'auto:run2_mc'}, stepNanoGen ])

steps['NANOEDMMC2018_PROD'] = merge([{'--conditions': 'auto:phase1_2018_realistic', '--era': 'Run2_2018', '--filein':'file:step3_inMINIAODSIM.root'}, stepNanoEDMMCProd ])
steps['NANOUP15'] = merge([{ '--conditions':'auto:run2_mc', '--era':'Run2_2016','-n':'10', '--filein':'file:step3_inMINIAODSIM.root','--nThreads':'2'}, stepNanoEDMMCProd ])
steps['NANOUP17'] = merge([{'--conditions':'auto:phase1_2017_realistic','--era': 'Run2_2017','-n':'10' ,'--filein':'file:step3_inMINIAODSIM.root', '--geometry':'DB:Extended', '--nThreads':'2'}, stepNanoEDMMCProd])
steps['NANOUP15Had']=merge([{'--filein':'file:step4_inMINIAODSIM.root'},steps['NANOUP15']])
steps['NANOUP15MC_PU25_JME']=merge([{'--customise':'PhysicsTools/NanoAOD/custom_jme_cff.PrepJMECustomNanoAOD_MC'},steps['NANOUP15']])
steps['NANOUP17Had']=merge([{'--filein':'file:step4_inMINIAODSIM.root'},steps['NANOUP17']])
steps['NANOUP18'] = merge([{'--conditions': 'auto:phase1_2018_realistic', '--era': 'Run2_2018','-n':'10', '--filein':'file:step3_inMINIAODSIM.root', '--nThreads':'2'}, stepNanoEDMMCProd ])
steps['NANOUP18Had']=merge([{'--filein':'file:step4_inMINIAODSIM.root'},steps['NANOUP18']])
steps['NANOPRODUP18']=merge([{'--filein':'file:step4.root'},steps['NANOUP18']])

steps['NANOUP15FS'] = merge([{'--filein':'file:step3.root','--fast':''}, steps['NANOUP15']])
steps['NANOUP17FS'] = merge([{'--filein':'file:step3.root','--fast':'','--era':'Run2_2017_FastSim'}, steps['NANOUP17']])
steps['NANOUP18FS'] = merge([{'--filein':'file:step3.root','--fast':'','--era':'Run2_2018_FastSim'}, steps['NANOUP18']])

steps['NANOUP15FSrefine'] = merge([{'--customise':'PhysicsTools/NanoAOD/jetsAK4_CHS_cff.nanoAOD_refineFastSim_bTagDeepFlav'}, steps['NANOUP15FS']])

steps['HEfail'] = {'--conditions':'auto:phase1_2018_realistic_HEfail',
                   '-n':'10',
                   '--pileup':'AVE_50_BX_25ns','--pileup_input':'das:/RelValMinBias_13/%s/GEN-SIM'%(baseDataSetRelease[18]),
                   '--era' : 'Run2_2018',
                   '--geometry' : 'DB:Extended'
                  }
steps['DigiFullHEfail']=merge([{'-s':'DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval2018','--datatier':'GEN-SIM-DIGI-RAW','--eventcontent':'FEVTDEBUGHLT'}, steps['HEfail']])
steps['RecoFullHEfail']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM',
                                '--datatier':'GEN-SIM-RECO,MINIAODSIM,DQMIO',
                                '--eventcontent':'RECOSIM,MINIAODSIM,DQM',
                               }, steps['HEfail']])
steps['HARVESTFullHEfail']=merge([{'-s':'HARVESTING:@standardValidationNoHLT+@standardDQMFakeHLT+@miniAODValidation+@miniAODDQM','--mc':'','--scenario' : 'pp','--filetype':'DQM'}, steps['HEfail']])
steps['NanoFullHEfail']={'-s':'NANO',
                         '--conditions':'auto:phase1_2018_realistic_HEfail',
                         '-n':'10',
                         '--era' : 'Run2_2018',
                         '--geometry' : 'DB:Extended',
                         '--datatier':'NANOAODSIM',
                         '--eventcontent':'NANOEDMAODSIM',
                         '--filein':'file:step3_inMINIAODSIM.root'}

steps['HcalNano']={'-s':'RAW2DIGI,RECO,USER:DPGAnalysis/HcalNanoAOD/hcalNano_cff.hcalNanoTask',
                         '--conditions':'auto:run3_data_prompt',
                         '-n':'10',
                         '--era' : 'Run3',
                         '--geometry' : 'DB:Extended',
                         '--datatier':'NANOAOD',
                         '--eventcontent':'NANOAOD'}

steps['HcalNanoCalibGap']={'-s':'RAW2DIGI,RECO,USER:DPGAnalysis/HcalNanoAOD/hcalNano_cff.hcalNanoTask',
                         '--conditions':'auto:run3_data_prompt',
                         '-n':'10',
                         '--era' : 'Run3',
                         '--geometry' : 'DB:Extended',
                         '--datatier':'NANOAOD',
                         '--eventcontent':'NANOAOD', 
                         '--customise':'DPGAnalysis/HcalNanoAOD/hcalNano_cff.customiseHcalCalib', 
                         '--processName': 'USER'}
                         
#################################################################################
####From this line till the end of the file :
####UPGRADE WORKFLOWS IN PREPARATION - Gaelle's sandbox -
#####Accessible only through the option --what upgrade
#####(unless specifically added to relval_2026.py)
#####Transparent for any of the standard workflows
#### list of worflows defined (not necessarly running though): runTheMatrix.py --what upgrade -n
####
###
#################################################################################

from  Configuration.PyReleaseValidation.upgradeWorkflowComponents import *

# imported from above, only non-empty values should be provided here
defaultDataSets['2017']='CMSSW_12_0_0_pre4-113X_mc2017_realistic_v5-v'
defaultDataSets['2017Design']='CMSSW_12_0_0_pre4-113X_mc2017_design_v5-v'
defaultDataSets['2018']='CMSSW_12_0_0_pre4-113X_upgrade2018_realistic_v5-v'
defaultDataSets['2018Design']='CMSSW_12_0_0_pre4-113X_upgrade2018_design_v5-v'
defaultDataSets['2021']='CMSSW_13_1_0_pre1-130X_mcRun3_2022_realistic_withNewBSFromEOY2022Data_v2_RV186-v'
defaultDataSets['2021Design']='CMSSW_12_5_0_pre4-124X_mcRun3_2022_design_v7_design_BS2022-v'
defaultDataSets['2021FS']='CMSSW_12_4_13-124X_mcRun3_2022_realistic_v12_2021_FastSim-v'
defaultDataSets['2023']='CMSSW_13_0_10-130X_mcRun3_2023_realistic_withEarly2023BS_v1_2023-v'
defaultDataSets['2023FS']='CMSSW_13_0_11-130X_mcRun3_2023_realistic_withEarly2023BS_v1_FastSim-v'
defaultDataSets['2024']='CMSSW_12_5_0_pre4-124X_mcRun3_2024_realistic_v11_BS2022-v'
defaultDataSets['2026D49']='CMSSW_12_0_0_pre4-113X_mcRun4_realistic_v7_2026D49noPU-v'
defaultDataSets['2026D76']='CMSSW_12_0_0_pre4-113X_mcRun4_realistic_v7_2026D76noPU-v'
defaultDataSets['2026D77']='CMSSW_12_1_0_pre2-113X_mcRun4_realistic_v7_2026D77noPU-v'
defaultDataSets['2026D88']='CMSSW_12_3_0_pre5-123X_mcRun4_realistic_v4_2026D88noPU-v'
defaultDataSets['2026D95']='CMSSW_13_1_0_pre1-130X_mcRun4_realistic_v2_2026D95noPU-v'
defaultDataSets['2026D96']='CMSSW_13_1_0_pre1-130X_mcRun4_realistic_v2_2026D96noPU-v'
defaultDataSets['2026D98']='CMSSW_13_2_0_pre1-131X_mcRun4_realistic_v5_2026D98noPU-v'

puDataSets = {}
for key, value in defaultDataSets.items(): puDataSets[key+'PU'] = value
defaultDataSets.update(puDataSets)

# sometimes v1 won't be used - override it here - the dictionary key is gen fragment + '_' + geometry
versionOverrides={'BuMixing_BMuonFilter_forSTEAM_13TeV_TuneCUETP8M1_2017':'2','HSCPstop_M_200_TuneCUETP8M1_13TeV_pythia8_2017':'2','RSGravitonToGammaGamma_kMpl01_M_3000_TuneCUETP8M1_13TeV_pythia8_2017':'2','WprimeToENu_M-2000_TuneCUETP8M1_13TeV-pythia8_2017':'2','DisplacedSUSY_stopToBottom_M_300_1000mm_TuneCUETP8M1_13TeV_pythia8_2017':'2','TenE_E_0_200_pythia8_2017':'2','TenE_E_0_200_pythia8_2017PU':'2', 'TenTau_E_15_500_pythia8_2018':'2','PhotonJet_Pt_10_13TeV_TuneCUETP8M1_2018':'2','Wjet_Pt_80_120_13TeV_TuneCUETP8M1_2018':'2','SingleMuPt10_Eta2p85_2021':'2','SingleMuPt100_Eta2p85_2021':'2','SingleMuPt1000_Eta2p85_2021':'2'}

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
    key='MinBias_14TeV_pythia8_TuneCP5'+'_'+ds
    name=baseDataSetReleaseBetter[key]
    if '2017' in ds:
        PUDataSets[ds]={'-n':10,'--pileup':'AVE_35_BX_25ns','--pileup_input':'das:/RelValMinBias_13/%s/GEN-SIM'%(name,)}
    elif '2018' in ds or 'postLS2' in ds:
        PUDataSets[ds]={'-n':10,'--pileup':'AVE_50_BX_25ns','--pileup_input':'das:/RelValMinBias_13/%s/GEN-SIM'%(name,)}
    elif '2021' in ds or '2023' in ds or '2024' in ds:
        if 'FS' not in ds:
            PUDataSets[ds]={'-n':10,'--pileup':'Run3_Flat55To75_PoissonOOTPU','--pileup_input':'das:/RelValMinBias_14TeV/%s/GEN-SIM'%(name,)}
        else:
            PUDataSets[ds]={'-n':10,'--pileup':'Run3_Flat55To75_PoissonOOTPU','--pileup_input':'das:/RelValMinBias_14TeV/%s/GEN-SIM-RECO'%(name,)}
    elif '2026' in ds:
        PUDataSets[ds]={'-n':10,'--pileup':'AVE_200_BX_25ns','--pileup_input':'das:/RelValMinBias_14TeV/%s/GEN-SIM'%(name,)}
    else:
        PUDataSets[ds]={'-n':10,'--pileup':'AVE_35_BX_25ns','--pileup_input':'das:/RelValMinBias_14TeV/%s/GEN-SIM'%(name,)}

    #PUDataSets[ds]={'-n':10,'--pileup':'AVE_50_BX_25ns','--pileup_input':'das:/RelValMinBias_13/%s/GEN-SIM'%(name,)}
    #PUDataSets[ds]={'-n':10,'--pileup':'AVE_70_BX_25ns','--pileup_input':'das:/RelValMinBias_13/%s/GEN-SIM'%(name,)}


upgradeStepDict={}
for specialType,specialWF in upgradeWFs.items():
	specialWF.init(upgradeStepDict)

# just make all combinations - yes, some will be nonsense.. but then these are not used unless specified above
# collapse upgradeKeys using list comprehension
for year,k in [(year,k) for year in upgradeKeys for k in upgradeKeys[year]]:
    k2=k
    if 'PU' in k[-2:]:
        k2=k[:-2]
    geom=upgradeProperties[year][k]['Geom']
    gt=upgradeProperties[year][k]['GT']
    hltversion=upgradeProperties[year][k].get('HLTmenu')
    beamspot=upgradeProperties[year][k].get('BeamSpot', None)

    # setup baseline steps
    upgradeStepDict['Gen'][k]= {'-s' : 'GEN',
                                '-n' : 10,
                                '--conditions' : gt,
                                '--beamspot' : 'Realistic25ns13TeVEarly2017Collision',
                                '--datatier' : 'GEN',
                                '--eventcontent': 'FEVTDEBUG',
                                '--geometry' : geom
                                }
    if beamspot is not None: upgradeStepDict['Gen'][k]['--beamspot']=beamspot

    upgradeStepDict['GenSim'][k]= {'-s' : 'GEN,SIM',
                                       '-n' : 10,
                                       '--conditions' : gt,
                                       '--beamspot' : 'Realistic25ns13TeVEarly2017Collision',
                                       '--datatier' : 'GEN-SIM',
                                       '--eventcontent': 'FEVTDEBUG',
                                       '--geometry' : geom
                                       }
    if beamspot is not None: upgradeStepDict['GenSim'][k]['--beamspot']=beamspot

    upgradeStepDict['GenSimHLBeamSpot'][k]= {'-s' : 'GEN,SIM',
                                       '-n' : 10,
                                       '--conditions' : gt,
                                       '--beamspot' : 'HLLHC',
                                       '--datatier' : 'GEN-SIM',
                                       '--eventcontent': 'FEVTDEBUG',
                                       '--geometry' : geom
                                       }

    upgradeStepDict['GenSimHLBeamSpot14'][k]= {'-s' : 'GEN,SIM',
                                       '-n' : 10,
                                       '--conditions' : gt,
                                       '--beamspot' : 'HLLHC14TeV',
                                       '--datatier' : 'GEN-SIM',
                                       '--eventcontent': 'FEVTDEBUG',
                                       '--geometry' : geom
                                       }

    upgradeStepDict['GenSimHLBeamSpotHGCALCloseBy'][k]= {'-s' : 'GEN,SIM',
                                       '-n' : 10,
                                       '--conditions' : gt,
                                       '--beamspot' : 'HGCALCloseBy',
                                       '--datatier' : 'GEN-SIM',
                                       '--eventcontent': 'FEVTDEBUG',
                                       '--geometry' : geom
                                       }

    upgradeStepDict['Digi'][k] = {'-s':'DIGI:pdigi_valid,L1,DIGI2RAW,HLT:%s'%(hltversion),
                                      '--conditions':gt,
                                      '--datatier':'GEN-SIM-DIGI-RAW',
                                      '-n':'10',
                                      '--eventcontent':'FEVTDEBUGHLT',
                                      '--geometry' : geom
                                      }

    # Adding Track trigger step in step2
    upgradeStepDict['DigiTrigger'][k] = {'-s':'DIGI:pdigi_valid,L1TrackTrigger,L1,DIGI2RAW,HLT:%s'%(hltversion),
                                      '--conditions':gt,
                                      '--datatier':'GEN-SIM-DIGI-RAW',
                                      '-n':'10',
                                      '--eventcontent':'FEVTDEBUGHLT',
                                      '--geometry' : geom
                                      }

    upgradeStepDict['HLTRun3'][k] = {'-s':'HLT:%s'%(hltversion),
                                 '--conditions':gt,
                                 '--datatier':'GEN-SIM-DIGI-RAW',
                                 '-n':'10',
                                 '--eventcontent':'FEVTDEBUGHLT',
                                 '--geometry' : geom,
                                 '--outputCommands' : '"drop *_*_*_GEN,drop *_*_*_DIGI2RAW"'
                                 }

    upgradeStepDict['HLT75e33'][k] = {'-s':'HLT:@relval2026',
                                      '--processName':'HLTX',
                                      '--conditions':gt,
                                      '--datatier':'FEVT',
                                      '-n':'10',
                                      '--eventcontent':'FEVTDEBUGHLT',
                                      '--geometry' : geom
                                      }

    upgradeStepDict['Reco'][k] = {'-s':'RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,VALIDATION:@standardValidation+@miniAODValidation,DQM:@standardDQM+@ExtraHLT+@miniAODDQM',
                                      '--conditions':gt,
                                      '--datatier':'GEN-SIM-RECO,MINIAODSIM,DQMIO',
                                      '-n':'10',
                                      '--eventcontent':'RECOSIM,MINIAODSIM,DQM',
                                      '--geometry' : geom
                                      }

    upgradeStepDict['RecoFakeHLT'][k] = {'-s':'RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM',
                                      '--conditions':gt,
                                      '--datatier':'GEN-SIM-RECO,MINIAODSIM,DQMIO',
                                      '-n':'10',
                                      '--eventcontent':'RECOSIM,MINIAODSIM,DQM',
                                      '--geometry' : geom
                                      }

    upgradeStepDict['RecoGlobal'][k] = {'-s':'RAW2DIGI,RECO,RECOSIM,PAT,VALIDATION:@phase2Validation+@miniAODValidation,DQM:@phase2+@miniAODDQM',
                                      '--conditions':gt,
                                      '--datatier':'GEN-SIM-RECO,MINIAODSIM,DQMIO',
                                      '-n':'10',
                                      '--eventcontent':'FEVTDEBUGHLT,MINIAODSIM,DQM',
                                      '--geometry' : geom
                                      }

    upgradeStepDict['RecoLocal'][k] = {'-s':'RAW2DIGI,L1Reco,RECO:localreco',
                                      '--conditions':gt,
                                      '--datatier':'GEN-SIM-RECO',
                                      '-n':'10',
                                      '--eventcontent':'FEVTDEBUGHLT',
                                      '--geometry' : geom
                                      }

    upgradeStepDict['MiniAOD'][k] = {'-s':'PAT',
                                      '--conditions':gt,
                                      '--datatier':'MINIAODSIM',
                                      '-n':'10',
                                      '--eventcontent':'MINIAODSIM',
                                      '--geometry' : geom
                                      }

    upgradeStepDict['HARVEST'][k]={'-s':'HARVESTING:@standardValidation+@standardDQM+@ExtraHLT+@miniAODValidation+@miniAODDQM',
                                    '--conditions':gt,
                                    '--mc':'',
                                    '--geometry' : geom,
                                    '--scenario' : 'pp',
                                    '--filetype':'DQM',
                                    }

    upgradeStepDict['HARVESTFakeHLT'][k]={'-s':'HARVESTING:@standardValidationNoHLT+@standardDQMFakeHLT+@miniAODValidation+@miniAODDQM',
                                          '--conditions':gt,
                                          '--mc':'',
                                          '--geometry' : geom,
                                          '--scenario' : 'pp',
                                          '--filetype':'DQM',
                                         }

    upgradeStepDict['HARVESTNano'][k]={'-s': 'HARVESTING:@standardValidation+@standardDQM+@ExtraHLT+@miniAODValidation+@miniAODDQM+@nanoAODDQM',
                                       '--conditions': gt,
                                       '--mc': '',
                                       '--geometry': geom,
                                       '--scenario': 'pp',
                                       '--filetype': 'DQM',
                                      }

    upgradeStepDict['HARVESTNanoFakeHLT'][k]={'-s': 'HARVESTING:@standardValidationNoHLT+@standardDQMFakeHLT+@miniAODValidation+@miniAODDQM+@nanoAODDQM',
                                              '--conditions': gt,
                                              '--mc': '',
                                              '--geometry': geom,
                                              '--scenario': 'pp',
                                              '--filetype': 'DQM',
                                             }

    upgradeStepDict['HARVESTGlobal'][k] = merge([{'-s': 'HARVESTING:@phase2Validation+@phase2+@miniAODValidation+@miniAODDQM'}, upgradeStepDict['HARVEST'][k]])

    upgradeStepDict['ALCA'][k] = {'-s':'ALCA:SiPixelCalSingleMuonLoose+SiPixelCalSingleMuonTight+TkAlMuonIsolated+TkAlMinBias+MuAlOverlaps+EcalESAlign+TkAlZMuMu+TkAlDiMuonAndVertex+HcalCalHBHEMuonProducerFilter+TkAlUpsilonMuMu+TkAlJpsiMuMu+SiStripCalMinBias',
                                      '--conditions':gt,
                                      '--datatier':'ALCARECO',
                                      '-n':'10',
                                      '--eventcontent':'ALCARECO',
                                      '--geometry' : geom,
                                      '--filein':'file:step3.root'
                                      }

    upgradeStepDict['ALCAPhase2'][k] = merge([{'-s':'ALCA:SiPixelCalSingleMuonLoose+SiPixelCalSingleMuonTight+TkAlMuonIsolated+TkAlMinBias+MuAlOverlaps+EcalESAlign+TkAlZMuMu+TkAlDiMuonAndVertex+HcalCalHBHEMuonProducerFilter+TkAlUpsilonMuMu+TkAlJpsiMuMu'},upgradeStepDict['ALCA'][k]])

    upgradeStepDict['FastSim'][k]={'-s':'GEN,SIM,RECO,VALIDATION',
                                   '--eventcontent':'FEVTDEBUGHLT,DQM',
                                   '--datatier':'GEN-SIM-DIGI-RECO,DQMIO',
                                   '--conditions':gt,
                                   '--fast':'',
                                   '--geometry' : geom,
                                   '--relval':'27000,3000'}

    upgradeStepDict['HARVESTFast'][k]={'-s':'HARVESTING:validationHarvesting',
                                    '--conditions':gt,
                                    '--mc':'',
                                    '--geometry' : geom,
                                    '--scenario' : 'pp'
                                    }

    upgradeStepDict['FastSimRun3'][k]={'-s':'SIM,RECOBEFMIX,DIGI:pdigi_valid,L1,DIGI2RAW,L1Reco,RECO,PAT,NANO,VALIDATION:@standardValidation+@miniAODValidation,DQM:@standardDQMFS+@miniAODDQM+@nanoAODDQM',
                                       '--fast':'',
                                       '--era':'Run3_FastSim',
                                       '--beamspot':beamspot,
                                       '--conditions':gt,
                                       '--geometry':geom,
                                       '--eventcontent':'FEVTDEBUGHLT,MINIAODSIM,NANOEDMAODSIM,DQM',
                                       '--datatier':'GEN-SIM-DIGI-RECO,MINIAODSIM,NANOAODSIM,DQMIO',
                                       }

    upgradeStepDict['HARVESTFastRun3'][k]={'-s':'HARVESTING:validationHarvesting+@miniAODValidation+@miniAODDQM+@nanoAODDQM',
                                           '--conditions':gt,
                                           '--mc':'',
                                           '--fast':'',
                                           '--geometry':geom,
                                           '--scenario':'pp',
                                           '--filetype':'DQM',
                                           '--filein':'file:step2_inDQM.root'}

    upgradeStepDict['Nano'][k] = {'-s':'NANO,DQM:@nanoAODDQM',
                                      '--conditions':gt,
                                      '--datatier':'NANOAODSIM,DQMIO',
                                      '-n':'10',
                                      '--eventcontent':'NANOEDMAODSIM,DQM',
                                      '--filein':'file:step3_inMINIAODSIM.root',
                                      '--geometry' : geom
                                      }

    upgradeStepDict['RecoNano'][k] = {'-s':'RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,NANO,VALIDATION:@standardValidation+@miniAODValidation,DQM:@standardDQM+@ExtraHLT+@miniAODDQM+@nanoAODDQM',
                                      '--conditions':gt,
                                      '--datatier':'GEN-SIM-RECO,MINIAODSIM,NANOAODSIM,DQMIO',
                                      '-n':'10',
                                      '--eventcontent':'RECOSIM,MINIAODSIM,NANOEDMAODSIM,DQM',
                                      '--geometry' : geom,
                                      }

    upgradeStepDict['RecoNanoFakeHLT'][k] = {'-s':'RAW2DIGI,L1Reco,RECO,RECOSIM,PAT,NANO,VALIDATION:@standardValidationNoHLT+@miniAODValidation,DQM:@standardDQMFakeHLT+@miniAODDQM+@nanoAODDQM',
                                             '--conditions':gt,
                                             '--datatier':'GEN-SIM-RECO,MINIAODSIM,NANOAODSIM,DQMIO',
                                             '-n':'10',
                                             '--eventcontent':'RECOSIM,MINIAODSIM,NANOEDMAODSIM,DQM',
                                             '--geometry' : geom,
                                            }

    # setup baseline and variations
    for specialType,specialWF in upgradeWFs.items():
        specialWF.setup(upgradeStepDict, k, upgradeProperties[year][k])

    # setup PU
    if k2 in PUDataSets:
        for specialType,specialWF in upgradeWFs.items():
            for step in specialWF.PU:
                stepName = specialWF.getStepName(step)
                stepNamePU = specialWF.getStepNamePU(step)
                if k not in upgradeStepDict[stepName] or upgradeStepDict[stepName][k] is None:
                    upgradeStepDict[stepNamePU][k] = None
                else:
                    upgradeStepDict[stepNamePU][k]=merge([PUDataSets[k2],upgradeStepDict[stepName][k]])

            # in case special WF has PU-specific changes: apply *after* basic PU step is created
            specialWF.setupPU(upgradeStepDict, k, upgradeProperties[year][k])

for step in upgradeStepDict.keys():
    # we need to do this for each fragment
    if ('Sim' in step and 'Fast' not in step) or ('Premix' in step) or ('Sim' not in step and 'Gen' in step):
        for frag,info in upgradeFragments.items():
            howMuch=info.howMuch
            for key in [key for year in upgradeKeys for key in upgradeKeys[year]]:
                k=frag[:-4]+'_'+key+'_'+step
                if step in upgradeStepDict and key in upgradeStepDict[step]:
                    if upgradeStepDict[step][key] is None:
                        steps[k]=None
                    elif 'Premix' in step:
                        # Include premixing stage1 only for SingleNu, use special step name
                        if not 'SingleNu' in frag:
                            continue
                        stepKey = 'PREMIX_'+key+'_'+step
                        howMuch = Kby(100,100)
                        steps[stepKey]=merge([ {'--evt_type':frag},howMuch,upgradeStepDict[step][key]])
                    else:
                        steps[k]=merge([ {'cfg':frag},howMuch,upgradeStepDict[step][key]])
                        #get inputs in case of -i...but no need to specify in great detail
                        #however, there can be a conflict of beam spots but this is lost in the dataset name
                        #so please be careful
                        s=frag[:-4]+'_'+key
                        # exclude upgradeKeys without input dataset, and special WFs that disable reuse
                        istep = step+preventReuseKeyword
                        if 'FastSim' not in k and s+'INPUT' not in steps and s in baseDataSetReleaseBetter and defaultDataSets[key] != '' and \
                           (istep not in upgradeStepDict or key not in upgradeStepDict[istep] or upgradeStepDict[istep][key] is not None):
                            if 'FS' not in key: #For FullSim
                                steps[k+'INPUT']={'INPUT':InputInfo(dataSet='/RelVal'+info.dataset+'/%s/GEN-SIM'%(baseDataSetReleaseBetter[s],),location='STD')}
                            else: #For FastSim to recycle GEN
                                steps[k+'INPUT']={'INPUT':InputInfo(dataSet='/RelVal'+info.dataset+'/%s/GEN'%(baseDataSetReleaseBetter[s],),location='STD')}
    else:
        for key in [key for year in upgradeKeys for key in upgradeKeys[year]]:
            k=step+'_'+key
            if step in upgradeStepDict and key in upgradeStepDict[step]:
                if upgradeStepDict[step][key] is None:
                    steps[k]=None
                else:
                    steps[k]=merge([upgradeStepDict[step][key]])
