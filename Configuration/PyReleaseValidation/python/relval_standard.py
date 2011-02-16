
class InputInfo(object):
    def __init__(self,label,dataSet,run=0,files=1000,events=2000000,location='CAF') :
        self.run = run
        self.files = files
        self.events = events
        self.location = location
        self.label = label
        self.dataSet = dataSet

#merge dictionnaries, with prioty on the [0] index
def merge(dict1list):
    import copy
    last=len(dict1list)-1
    if last==0:
        return copy.copy(dict1list[0])
    else:
        reducedlist=dict1list[0:max(0,last-2)]
        #make a copy of the last item
        d=copy.copy(dict1list[last])
        #update with the last but one item
        d.update(dict1list[last-1])
        #and recursively do the rest
        reducedlist.append(d)
        return merge(reducedlist)


# step1 gensim
step1Defaults = {'--relval'      : None, # need to be explicitly set
                 '-s'            : 'GEN,SIM',
                 '-n'            : 10,
                 '--conditions'  : 'auto:mc',
                 '--datatier'    : 'GEN-SIM',
                 '--eventcontent': 'RAWSIM',
                 }


step1 = {}

#### Production test section ####
step1['ProdMinBias']=merge([{'cfg':'MinBias_7TeV_cfi','--relval':'9000,100'},step1Defaults])
step1['ProdMinBiasINPUT']={'INPUT':InputInfo(dataSet='/RelValProdMinBias/CMSSW_4_2_0_pre2-MC_311_V1-v1/GEN-SIM-RAW',label='prodmbrv',location='STD')}

step1['ProdTTbar']=merge([{'cfg':'TTbar_Tauola_7TeV_cfi','--relval':'9000,50'},step1Defaults])
step1['ProdTTbarINPUT']={'INPUT':InputInfo(dataSet='/RelValProdTTbar/CMSSW_4_2_0_pre2-MC_311_V1-v1/GEN-SIM-RAW',label='prodttbrv',location='STD')}

step1['ProdQCD_Pt_3000_3500']=merge([{'cfg':'QCD_Pt_3000_3500_7TeV_cfi','--relval':'9000,25'},step1Defaults])
step1['ProdQCD_Pt_3000_3500INPUT']={'INPUT':InputInfo(dataSet='/RelValProdQCD_Pt_3000_3500/CMSSW_4_2_0_pre2-MC_311_V1-v1/GEN-SIM-RAW',label='qcd335',location='STD')}



#### data ####
step1['RunCosmicsA']={'INPUT':InputInfo(dataSet='/Cosmics/Run2010A-v1/RAW',label='cos2010A',run=142089,events=100000)}
step1['MinimumBias2010A']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2010A-valskim-v6/RAW-RECO',label='run2010A',location='STD')}
step1['MinimumBias2010B']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2010B-valskim-v2/RAW-RECO',label='run2010B')}
step1['WZMuSkim2010A']={'INPUT':InputInfo(dataSet='/Mu/Run2010A-WZMu-Nov4Skim_v1/RAW-RECO',label='wzMu2010A')}
step1['WZMuSkim2010B']={'INPUT':InputInfo(dataSet='/Mu/Run2010B-WZMu-Nov4Skim_v1/RAW-RECO',label='wzMu2010B')}
step1['WZEGSkim2010A']={'INPUT':InputInfo(dataSet='/EG/Run2010A-WZEG-Nov4Skim_v1/RAW-RECO',label='wzEG2010A')}
step1['WZEGSkim2010B']={'INPUT':InputInfo(dataSet='/Electron/Run2010B-WZEG-Nov4Skim_v1/RAW-RECO',label='wzEG2010B')}

step1['RunMinBias2010B']={'INPUT':InputInfo(dataSet='/MinimumBias/Run2010B-v1/RAW',label='mb2010B',run=149011,events=100000)}
step1['RunMu2010B']={'INPUT':InputInfo(dataSet='/Mu/Run2010B-v1/RAW',label='mu2010B',run=149011,events=100000)}
step1['RunElectron2010B']={'INPUT':InputInfo(dataSet='/Electron/Run2010B-v1/RAW',label='electron2010B',run=149011,events=100000)}
step1['RunPhoton2010B']={'INPUT':InputInfo(dataSet='/Photon/Run2010B-v1/RAW',label='photon2010B',run=149011,events=100000)}
step1['RunJet2010B']={'INPUT':InputInfo(dataSet='/Jet/Run2010B-v1/RAW',label='jet2010B',run=149011,events=100000)}

#### Standard release validation samples ####

stCond={'--conditions':'auto:startup'}
K9by25={'--relval':'9000,25'}
K9by50={'--relval':'9000,50'}
K9by100={'--relval':'9000,100'}
K9by250={'--relval':'9000,250'}
   
step1['MinBias']=merge([{'cfg':'MinBias_7TeV_cfi'},K9by100,step1Defaults])
step1['QCD_Pt_3000_3500']=merge([{'cfg':'QCD_Pt_3000_3500_7TeV_cfi'},K9by25,step1Defaults])
step1['QCD_Pt_80_120']=merge([{'cfg':'QCD_Pt_80_120_7TeV_cfi'},K9by50,step1Defaults])
step1['SingleElectronPt10']=merge([{'cfg':'SingleElectronPt10_cfi'},K9by250,step1Defaults])
step1['SingleElectronPt35']=merge([{'cfg':'SingleElectronPt35_cfi'},K9by250,step1Defaults])
step1['SingleGammaPt10']=merge([{'cfg':'SingleGammaPt10_cfi'},K9by100,step1Defaults])
step1['SingleGammaPt35']=merge([{'cfg':'SingleGammaPt35_cfi'},K9by100,step1Defaults])
step1['SingleMuPt10']=merge([{'cfg':'SingleMuPt10_cfi','--relval':'25000,250'},step1Defaults])
step1['SingleMuPt100']=merge([{'cfg':'SingleMuPt100_cfi','--relval':'9000,250'},step1Defaults])
step1['SingleMuPt1000']=merge([{'cfg':'SingleMuPt1000_cfi'},K9by100,step1Defaults])
step1['TTbar']=merge([{'cfg':'TTbar_Tauola_7TeV_cfi'},K9by50,step1Defaults])
step1['ZEE']=merge([{'cfg':'ZEE_7TeV_cfi'},K9by100,step1Defaults])
step1['Wjet_Pt_80_120']=merge([{'cfg':'Wjet_Pt_80_120_7TeV_cfi'},K9by100,step1Defaults])
step1['Wjet_Pt_3000_3500']=merge([{'cfg':'Wjet_Pt_3000_3500_7TeV_cfi'},K9by100,step1Defaults])
step1['LM1_sfts']=merge([{'cfg':'LM1_sfts_7TeV_cfi'},K9by100,step1Defaults])
step1['QCD_FlatPt_15_3000']=merge([{'cfg':'QCDForPF_7TeV_cfi'},K9by100,step1Defaults])

step1['MinBiasINPUT']={'INPUT':InputInfo(dataSet='/RelValMinBias/CMSSW_4_2_0_pre2-MC_311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='MinBiasrv',location='STD')}
step1['QCD_Pt_3000_3500INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_3000_3500/CMSSW_4_2_0_pre2-MC_311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='QCD_Pt_3000_3500rv',location='STD')}
step1['QCD_Pt_80_120INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_80_120/CMSSW_4_2_0_pre2-MC_311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='QCD_Pt_80_120rv',location='STD')}
step1['SingleElectronPt10INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleElectronPt10/CMSSW_4_2_0_pre2-MC_311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='SingleElectronPt10rv',location='STD')}
step1['SingleElectronPt35INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleElectronPt35/CMSSW_4_2_0_pre2-MC_311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='SingleElectronPt35rv',location='STD')}
step1['SingleGammaPt10INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleGammaPt10/CMSSW_4_2_0_pre2-MC_311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='SingleGammaPt10rv',location='STD')}
step1['SingleGammaPt35INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleGammaPt35/CMSSW_4_2_0_pre2-MC_311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='SingleGammaPt35rv',location='STD')}
step1['SingleMuPt10INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt10/CMSSW_4_2_0_pre2-MC_311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='SingleMuPt10rv',location='STD')}
step1['SingleMuPt100INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt100/CMSSW_4_2_0_pre2-MC_311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='SingleMuPt100rv',location='STD')}
step1['SingleMuPt1000INPUT']={'INPUT':InputInfo(dataSet='/RelValSingleMuPt1000/CMSSW_4_2_0_pre2-MC_311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='SingleMuPt1000rv',location='STD')}
step1['TTbarINPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/CMSSW_4_2_0_pre2-MC_311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='ttbarrv',location='STD')}
step1['ZEEINPUT']={'INPUT':InputInfo(dataSet='/RelValZEE/CMSSW_4_2_0_pre2-MC_311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='ZEErv',location='STD')}
step1['Wjet_Pt_80_120INPUT']={'INPUT':InputInfo(dataSet='/RelValWjet_Pt_80_120/CMSSW_4_2_0_pre2-MC_311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='Wjet_Pt_80_120rv',location='STD')}
step1['Wjet_Pt_3000_3500INPUT']={'INPUT':InputInfo(dataSet='/RelValWjet_Pt_3000_3500/CMSSW_4_2_0_pre2-MC_311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='Wjet_Pt_3000_3500rv',location='STD')}
step1['LM1_sftsINPUT']={'INPUT':InputInfo(dataSet='/RelValLM1_sfts/CMSSW_4_2_0_pre2-MC_311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='LM1_sftsrv',location='STD')}
step1['QCD_FlatPt_15_3000INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_FlatPt_15_3000/CMSSW_4_2_0_pre2-MC_311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='QCD_FlatPt_15_3000rv',location='STD')}

   
step1['MinBias2']=merge([{'cfg':'MinBias_7TeV_cfi'},stCond,K9by100,step1Defaults])
step1['Higgs200ChargedTaus']=merge([{'cfg':'H200ChargedTaus_Tauola_7TeV_cfi'},stCond,K9by100,step1Defaults])
step1['QCD_Pt_3000_3500_2']=merge([{'cfg':'QCD_Pt_3000_3500_7TeV_cfi'},K9by25,stCond,step1Defaults])
step1['QCD_Pt_80_120_2']=merge([{'cfg':'QCD_Pt_80_120_7TeV_cfi'},K9by50,stCond,step1Defaults])
step1['JpsiMM']=merge([{'cfg':'JpsiMM_7TeV_cfi','--relval':'65250,725'},stCond,step1Defaults])
step1['TTbar2']=merge([{'cfg':'TTbar_Tauola_7TeV_cfi'},K9by50,stCond,step1Defaults])
step1['WE']=merge([{'cfg':'WE_7TeV_cfi'},K9by100,stCond,step1Defaults])
step1['WM']=merge([{'cfg':'WM_7TeV_cfi'},K9by100,stCond,step1Defaults])
step1['ZEE2']=merge([{'cfg':'ZEE_7TeV_cfi'},K9by100,stCond,step1Defaults])
step1['ZMM']=merge([{'cfg':'ZMM_7TeV_cfi','--relval':'18000,200'},stCond,step1Defaults])
step1['ZTT']=merge([{'cfg':'ZTT_Tauola_All_hadronic_7TeV_cfi'},K9by100,stCond,step1Defaults])
step1['H130GGgluonfusion']=merge([{'cfg':'H130GGgluonfusion_7TeV_cfi'},K9by100,stCond,step1Defaults])
step1['PhotonJets_Pt_10']=merge([{'cfg':'PhotonJet_Pt_10_7TeV_cfi'},K9by100,stCond,step1Defaults])
step1['QQH1352T_Tauola']=merge([{'cfg':'QQH1352T_Tauola_7TeV_cfi'},K9by100,stCond,step1Defaults])

step1['MinBias2INPUT']={'INPUT':InputInfo(dataSet='/RelValMinBias/CMSSW_4_2_0_pre2-START311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='MinBiasrv',location='STD')}
step1['Higgs200ChargedTausINPUT']={'INPUT':InputInfo(dataSet='/RelValHiggs200ChargedTaus/CMSSW_4_2_0_pre2-START311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='Higgs200ChargedTausrv',location='STD')}
step1['QCD_Pt_3000_3500_2INPUT']={'INPUT':InputInfo(dataSet='/RelValQCD_Pt_3000_3500/CMSSW_4_2_0_pre2-START311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='QCD_Pt_3000_3500rv',location='STD')}
step1['JpsiMMINPUT']={'INPUT':InputInfo(dataSet='/RelValJpsiMM/CMSSW_4_2_0_pre2-START311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='JpsiMMrv',location='STD')}
step1['TTbar2INPUT']={'INPUT':InputInfo(dataSet='/RelValTTbar/CMSSW_4_2_0_pre2-START311_V0-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='ttbarrv',location='STD')}
step1['WEINPUT']={'INPUT':InputInfo(dataSet='/RelValWE/CMSSW_4_2_0_pre2-START311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='WErv',location='STD')}
step1['ZEE2INPUT']={'INPUT':InputInfo(dataSet='/RelValZEE/CMSSW_4_2_0_pre2-START311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='ZEErv',location='STD')}
step1['ZTTINPUT']={'INPUT':InputInfo(dataSet='/RelValZTT/CMSSW_4_2_0_pre2-START311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='ZTTrv',location='STD')}
step1['H130GGgluonfusionINPUT']={'INPUT':InputInfo(dataSet='/RelValH130GGgluonfusion/CMSSW_4_2_0_pre2-START311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='H130GGgluonfusionrv',location='STD')}
step1['PhotonJets_Pt_10INPUT']={'INPUT':InputInfo(dataSet='/RelValPhotonJets_Pt_10/CMSSW_4_2_0_pre2-START311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='PhotonJets_Pt_10rv',location='STD')}
step1['QQH1352T_TauolaINPUT']={'INPUT':InputInfo(dataSet='/RelValQQH1352T_Tauola_cfi/CMSSW_4_2_0_pre2-START311_V1-v1/GEN-SIM-DIGI-RAW-HLTDEBUG',label='QQH1352T_Tauola_cfirv',location='STD')}

step1['Cosmics']=merge([{'cfg':'UndergroundCosmicMu_cfi.py','--relval':'666000,7400','--scenario':'cosmics'},step1Defaults])
step1['BeamHalo']=merge([{'cfg':'BeamHalo_cfi.py','--scenario':'cosmics'},K9by100,step1Defaults])




#### fastsim section ####
##no forseen to do things in two steps GEN-SIM then FASTIM->end: maybe later
step1FastDefaults = {'-s':'GEN,FASTSIM,HLT:GRun,VALIDATION', '--eventcontent':'FEVTDEBUGHLT','--datatier':'GEN-SIM-DIGI-RECO','--relval':'27000,1000'}
K100byK1={'--relval':'100000,1000'}
step1['TTbarFS1']=merge([{'cfg':'TTbar_Tauola_7TeV_cfi'},K100byK1,step1FastDefaults])
step1['TTbarFS2']=merge([{'cfg':'TTbar_Tauola_7TeV_cfi'},K100byK1,stCond,step1FastDefaults])
step1['SingleMuPt10FS']=merge([{'cfg':'SingleMuPt10_cfi'},step1FastDefaults])
step1['SingleMuPt100FS']=merge([{'cfg':'SingleMuPt100_cfi'},step1FastDefaults])
step1['ZEEFS1']=merge([{'cfg':'ZEE_7TeV_cfi'},K100byK1,step1FastDefaults])
step1['ZEEFS2']=merge([{'cfg':'ZEE_7TeV_cfi'},K100byK1,stCond,step1FastDefaults])
step1['QCDFlatPt153000FS']=merge([{'cfg':'QCDForPF_7TeV_cfi'},step1FastDefaults])
step1['H130GGgluonfusionFS']=merge([{'cfg':'H130GGgluonfusion_7TeV_cfi'},step1FastDefaults])
##########################



# step2 
step2Defaults = { 'cfg'           : 'step2',
                  '-s'            : 'DIGI,L1,DIGI2RAW,HLT:GRun,RAW2DIGI,L1Reco',
                  '--datatier'    : 'GEN-SIM-DIGI-RAW-HLTDEBUG',
                  '--eventcontent': 'FEVTDEBUGHLT',
                  '--conditions'  : 'auto:mc',
                  }

step2 = {}

step2['DIGIPROD1']=merge([{'--eventcontent':'RAWSIM','--datatier':'GEN-SIM-RAW'},step2Defaults])
step2['DIGI1']=merge([step2Defaults])
step2['DIGI2']=merge([stCond,step2Defaults])
step2['DIGICOS']=merge([{'--scenario':'cosmics'},stCond,step2Defaults])

dataReco={'--conditions':'auto:com10',
          '-s':'RAW2DIGI,L1Reco,RECO,ALCA:SiStripCalZeroBias+SiStripCalMinBias,DQM',
          '--datatier':'RECO,DQM',
          '--eventcontent':'RECO,DQM',
          '--data':'',
          '--magField':'AutoFromDBCurrent',
          '--customise':'Configuration/DataProcessing/RecoTLR.customisePPData',
          '--inputCommands':'"keep *","drop *_*_*_RECO"',
          '--process':'reRECO'
          }
step2['RECOD']=merge([dataReco])
step2['RECOVALSKIM']=merge([{'--customise':'Configuration/DataProcessing/RecoTLR.customiseVALSKIM','-s':'RAW2DIGI,L1Reco,RECO,DQM'},step2['RECOD']])
step2['RECOVALSKIMALCA']=merge([{'--customise':'Configuration/DataProcessing/RecoTLR.customiseVALSKIM'},step2['RECOD']])


step2['RECOCOSD']=merge([{'--scenario':'cosmics',
                          '-s':'RAW2DIGI,L1Reco,RECO,L1HwVal,DQM,ALCA:MuAlCalIsolatedMu+DtCalib',
                          '--customise':'Configuration/DataProcessing/RecoTLR.customiseCosmicData'
                          },dataReco])

# step3 
step3Defaults = { 'cfg'           : 'step3',
                  '-s'            : 'RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                  '--filein'      : 'file:reco.root',
                  '--conditions'  : 'auto:mc',
                  '--no_exec'     : '',
                  '--datatier'    : 'GEN-SIM-RECO',
                  '--eventcontent': 'RECOSIM'
                  }

step3 = {}

step3['RECO1']=merge([step3Defaults])
step3['RECO2']=merge([stCond,step3Defaults])
step3['RECOPROD1']=merge([{ '-s' : 'RAW2DIGI,L1Reco,RECO', '--datatier' : 'GEN-SIM-RECO,AODSIM', '--eventcontent' : 'RECOSIM,AODSIM'},step3Defaults])
step3['RECOMU']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCA:MuAlCalIsolatedMu+DtCalib'},stCond,step3Defaults])
step3['RECOCOS']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCA:MuAlCalIsolatedMu','--scenario':'cosmics'},stCond,step3Defaults])
step3['RECOMIN']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCA:SiStripCalZeroBias+SiStripCalMinBias+EcalCalPhiSym+EcalCalPi0Calib+EcalCalEtaCalib,VALIDATION,DQM'},stCond,step3Defaults])
step3['RECOQCD']=merge([{'-s':'RAW2DIGI,L1Reco,RECO,ALCA:MuAlCalIsolatedMu+DtCalib+EcalCalPi0Calib+EcalCalEtaCalib,VALIDATION,DQM'},stCond,step3Defaults])

step3['ALCACOSD']={'--conditions':'auto:com10',
                   '--datatier':'ALCARECO',
                   '--eventcontent':'ALCARECO',
                   '--scenario':'cosmics',
                   '-s':'ALCA:TkAlCosmics0T+MuAlGlobalCosmics+HcalCalHOCosmics+DQM'
                   }

   
# nothing defined yet, but keep in list for compatibility
step4Defaults = { 'cfg'           : 'step4',
                  '-s'            : 'ALCA:TkAlMuonIsolated+TkAlMinBias+EcalCalElectron+HcalCalIsoTrk+MuAlOverlaps',
                  '-n'            : 1000,
                  '--filein'      : 'file:reco.root',
                  '--conditions'  : 'auto:mc',
                  '--no_exec'     : '',
                  '--datatier'    : 'ALCARECO',
                  '--oneoutput'   : '',
                  '--eventcontent': 'ALCARECO',
                  }
step4 = {}

step4['ALCATT1']=merge([step4Defaults])
step4['ALCATT2']=merge([stCond,step4Defaults])
step4['ALCAMIN']=merge([{'-s':'ALCA:TkAlMinBias'},stCond,step4Defaults])
step4['ALCAQCD']=merge([{'-s':'ALCA:HcalCalIsoTrk+HcalCalDijets+HcalCalHO'},stCond,step4Defaults])
step4['ALCAMU']=merge([{'-s':'ALCA:MuAlOverlaps+TkAlMuonIsolated+TkAlZMuMu'},stCond,step4Defaults])
step4['ALCACOS']=merge([{'-s':'ALCA:TkAlCosmics0T+MuAlGlobalCosmics+HcalCalHOCosmics'},stCond,step4Defaults])
step4['ALCABH']=merge([{'-s':'ALCA:TkAlBeamHalo+MuAlBeamHaloOverlaps+MuAlBeamHalo'},stCond,step4Defaults])

stepList = [step1, step2, step3, step4]





# finally define the workflows as a combination of the above:
workflows = {}

## production tests
workflows[1]=['ProdMinBias','DIGIPROD1','RECOPROD1']
workflows[2]=['ProdTTbar','DIGIPROD1','RECOPROD1']
workflows[3]=['ProdQCD_Pt_3000_3500','DIGIPROD1','RECOPROD1']

### data ###
workflows[4.5]=['RunCosmicsA','RECOCOSD','ALCACOSD']
workflows[4.45]=['RunCosmicsA','RECOD']
workflows[4.6]=['MinimumBias2010A','RECOVALSKIM']
workflows[4.7]=['MinimumBias2010B','RECOVALSKIMALCA']
workflows[4.8]=['WZMuSkim2010A','RECOVALSKIM']
workflows[4.9]=['WZEGSkim2010A','RECOVALSKIM']
workflows[4.10]=['WZMuSkim2010B','RECOVALSKIM']
workflows[4.11]=['WZEGSkim2010B','RECOVALSKIM']
workflows[4.12]=['RunMinBias2010B','RECOD']
workflows[4.13]=['RunMu2010B','RECOD']
workflows[4.14]=['RunElectron2010B','RECOD']
workflows[4.15]=['RunPhoton2010B','RECOD']
workflows[4.16]=['RunJet2010B','RECOD']

### fastsim ###
workflows[5.1]= ['TTbarFS1']
workflows[6.3]= ['TTbarFS2']
workflows[5.2]= ['SingleMuPt10FS']
workflows[5.3]= ['SingleMuPt100FS']
workflows[6.1]= ['ZEEFS1']
workflows[6.2]= ['ZEEFS2']
workflows[39] = ['QCDFlatPt153000FS']
workflows[6.4] = ['H130GGgluonfusionFS']

### standard set ###
workflows[10]=['MinBias','DIGI1','RECO1']
workflows[12]=['QCD_Pt_3000_3500','DIGI1','RECO1']
workflows[14]=['QCD_Pt_80_120','DIGI1','RECO1']
workflows[16]=['SingleElectronPt10','DIGI1','RECO1']
workflows[17]=['SingleElectronPt35','DIGI1','RECO1']
workflows[18]=['SingleGammaPt10','DIGI1','RECO1']
workflows[19]=['SingleGammaPt35','DIGI1','RECO1']
workflows[20]=['SingleMuPt10','DIGI1','RECO1']
workflows[21]=['SingleMuPt100','DIGI1','RECO1']
workflows[22]=['SingleMuPt1000','DIGI1','RECO1']
workflows[24]=['TTbar','DIGI1','RECO1']
workflows[28]=['ZEE','DIGI1','RECO1']
workflows[35]=['Wjet_Pt_80_120','DIGI1','RECO1']
workflows[36]=['Wjet_Pt_3000_3500','DIGI1','RECO1']
workflows[37]=['LM1_sfts','DIGI1','RECO1']
workflows[38]=['QCD_FlatPt_15_3000','DIGI1','RECO1']

workflows[9]=['Higgs200ChargedTaus','DIGI2','RECO2']
workflows[13]=['QCD_Pt_3000_3500_2','DIGI2','RECO2']
workflows[23]=['JpsiMM','DIGI2','RECO2']
workflows[25]=['TTbar2','DIGI2','RECO2','ALCATT2']
workflows[26]=['WE','DIGI2','RECO2']
workflows[29]=['ZEE2','DIGI2','RECO2']
workflows[31]=['ZTT','DIGI2','RECO2']
workflows[32]=['H130GGgluonfusion','DIGI2','RECO2']
workflows[33]=['PhotonJets_Pt_10','DIGI2','RECO2']
workflows[34]=['QQH1352T_Tauola','DIGI2','RECO2']


workflows[7]=['Cosmics','DIGICOS','RECOCOS','ALCACOS']
workflows[8]=['BeamHalo','DIGICOS','RECOCOS','ALCABH']
workflows[11]=['MinBias2','DIGI2','RECOMIN','ALCAMIN']
workflows[15]=['QCD_Pt_80_120_2','DIGI1','RECOQCD','ALCAQCD']
workflows[27]=['WM','DIGI2','RECOMU','ALCAMU']
workflows[30]=['ZMM','DIGI2','RECOMU','ALCAMU']


### HI test ###
#workflows[40]
#workflows[41]

#### special wf ###
step1['TTbar_REDIGI_RERECO']=merge([{'cfg':'TTbar_Tauola_7TeV_cfi',
                                     '-s':'GEN,SIM,DIGI,L1,DIGI2RAW,HLT:GRun,RAW2DIGI,L1Reco,RECO,ALCA:MuAlCalIsolatedMu+DtCalib,VALIDATION,DQM',
                                     '--datatier':'GEN-SIM-DIGI-RAW-HLTDEBUG-RECO',
                                     '--eventcontent':'FEVTDEBUGHLT'},
                                    K9by50,stCond,step1Defaults])
step2['REDIGI2RECO']=merge([{'-s':'DIGI,L1,DIGI2RAW,HLT:GRun,RAW2DIGI,L1Reco,RECO,VALIDATION,DQM',
                             '--customise':'Configuration/StandardSequences/DigiToRecoNoPU.customise',
                             '--filtername':'REDIGItoRECO',
                             '--process':'REDIGI'},
                            stCond,step3Defaults])
step3['RECOFROMRECO']=merge([{'-s':'RECO,ALCA:MuAlCalIsolatedMu+DtCalib',
                              '--filtername':'RECOfromRECO'},
                             stCond,step3Defaults])
workflows[42]=['TTbar_REDIGI_RERECO','REDIGI2RECO','RECOFROMRECO']

