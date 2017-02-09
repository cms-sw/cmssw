from copy import deepcopy

# DON'T CHANGE THE ORDER, only append new keys. Otherwise the numbering for the runTheMatrix tests will change.

upgradeKeys = {}

upgradeKeys[2017] = [
    '2017',
    '2017PU',
    '2017Design',
    '2017DesignPU',
    '2018',
#    '2018PU',
    '2018Design',
#    '2018DesignPU',
]

upgradeKeys[2023] = [
    '2023D7',
    '2023D7PU',
    '2023D10',
    '2023D10PU',    
    '2023D4',
    '2023D4PU',
    '2023D7Timing',
    '2023D7TimingPU',
    '2023D10Timing',
    '2023D10TimingPU',
    '2023D4Timing',
    '2023D4TimingPU',
    '2023D8',
    '2023D8PU',
    '2023D9',
    '2023D9PU',
    '2023D11',
    '2023D11PU'
    
]

# pre-generation of WF numbers
numWFStart={
    2017: 10000,
    2023: 20000,
}
numWFSkip=200
# first two sets are the former D3 WF (now removed as redundant)
# temporary measure to keep other WF numbers the same
numWFConflict = [[11000,11200],[11400,11600],[20800,21200],[22400,22800],[25000,26000],[50000,51000]]
numWFAll={
    2017: [numWFStart[2017]],
    2023: [numWFStart[2023]]
}

for year in upgradeKeys:
    for i in range(1,len(upgradeKeys[year])):
        numWFtmp = numWFAll[year][i-1] + numWFSkip
        for conflict in numWFConflict:
            if numWFtmp>=conflict[0] and numWFtmp<conflict[1]:
                numWFtmp = conflict[1]
                break
        numWFAll[year].append(numWFtmp)

upgradeSteps=[
    'GenSimFull',
    'GenSimHLBeamSpotFull',
    'GenSimHLBeamSpotFull14',
    'DigiFull',
    'DigiFullTrigger',
    'DigiFullTriggerPU',
    'RecoFullLocal',
    'RecoFullLocalPU',
    'RecoFull',
    'RecoFullGlobal',
    'RecoFullGlobalPU',
    'HARVESTFull',
    'FastSim',
    'HARVESTFast',
    'DigiFullPU',
    'RecoFullPU',
    'HARVESTFullPU',
    'RecoFull_trackingOnly',
    'RecoFull_trackingOnlyPU',
    'HARVESTFull_trackingOnly',
    'HARVESTFull_trackingOnlyPU',
    'HARVESTFullGlobal',
    'HARVESTFullGlobalPU',
    'ALCAFull'
]

upgradeProperties = {}

upgradeProperties[2017] = {
    '2017' : {
        'Geom' : 'Extended2017Plan1',
        'GT' : 'auto:phase1_2017_realistic',
        'HLTmenu': '@relval2016',
        'Era' : 'Run2_2017',
        'Custom' : 'SLHCUpgradeSimulations/Configuration/HCalCustoms.customise_Hcal2017Plan1',
        'ScenToRun' : ['GenSimFull','DigiFull','RecoFull','ALCAFull','HARVESTFull'],
    },
    '2017Design' : {
        'Geom' : 'Extended2017Plan1',
        'GT' : 'auto:phase1_2017_design',
        'HLTmenu': '@relval2016',
        'Era' : 'Run2_2017',
        'Custom' : 'SLHCUpgradeSimulations/Configuration/HCalCustoms.customise_Hcal2017Plan1',
        'BeamSpot': 'GaussSigmaZ4cm',
        'ScenToRun' : ['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
    },
    '2018' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2018_realistic',
        'HLTmenu': '@relval2016',
        'Era' : 'Run2_2018',
        'ScenToRun' : ['GenSimFull','DigiFull','RecoFull','ALCAFull','HARVESTFull'],
    },
    '2018Design' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2018_design',
        'HLTmenu': '@relval2016',
        'Era' : 'Run2_2018',
        'BeamSpot': 'GaussSigmaZ4cm',
        'ScenToRun' : ['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
    },
}

upgradeProperties[2017]['2017PU'] = deepcopy(upgradeProperties[2017]['2017'])
upgradeProperties[2017]['2017PU']['ScenToRun'] = ['GenSimFull','DigiFullPU','RecoFullPU','HARVESTFullPU']
upgradeProperties[2017]['2017DesignPU'] = deepcopy(upgradeProperties[2017]['2017Design'])
upgradeProperties[2017]['2017DesignPU']['ScenToRun'] = ['GenSimFull','DigiFullPU','RecoFullPU','HARVESTFullPU']
upgradeProperties[2017]['2018PU'] = deepcopy(upgradeProperties[2017]['2018'])
upgradeProperties[2017]['2018PU']['ScenToRun'] = ['GenSimFull','DigiFullPU','RecoFullPU','HARVESTFullPU']
upgradeProperties[2017]['2018DesignPU'] = deepcopy(upgradeProperties[2017]['2018Design'])
upgradeProperties[2017]['2018DesignPU']['ScenToRun'] = ['GenSimFull','DigiFullPU','RecoFullPU','HARVESTFullPU']

upgradeProperties[2023] = {
    '2023D7' : {
        'Geom' : 'Extended2023D7',
        'GT' : 'auto:phase2_realistic',
        'HLTmenu': '@fake',
        'Era' : 'Phase2C1',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal','HARVESTFullGlobal'],
    },
    '2023D10' : {
        'Geom' : 'Extended2023D10',
        'GT' : 'auto:phase2_realistic',
        'HLTmenu': '@fake',
        'Era' : 'Phase2C1',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal','HARVESTFullGlobal'],
    },
    '2023D4' : {
        'Geom' : 'Extended2023D4',
        'HLTmenu': '@fake',
        'GT' : 'auto:phase2_realistic',
        'Era' : 'Phase2C2',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
    },
    '2023D8' : {
        'Geom' : 'Extended2023D8',
        'HLTmenu': '@fake',
        'GT' : 'auto:phase2_realistic',
        'Era' : 'Phase2C2_timing_layer',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
    },
    '2023D9' : {
        'Geom' : 'Extended2023D9',
        'GT' : 'auto:phase2_realistic',
        'HLTmenu': '@fake',
        'Era' : 'Phase2C1',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
    },
    '2023D11' : {
        'Geom' : 'Extended2023D11',
        'HLTmenu': '@fake',
        'GT' : 'auto:phase2_realistic',
        'Era' : 'Phase2C2',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
    }

}



#Timing (later we can alter geometry, etc, if need be)
upgradeProperties[2023]['2023D7Timing'] = deepcopy(upgradeProperties[2023]['2023D7'])
upgradeProperties[2023]['2023D7Timing']['Era'] = 'Phase2C1_timing'
upgradeProperties[2023]['2023D10Timing'] = deepcopy(upgradeProperties[2023]['2023D10'])
upgradeProperties[2023]['2023D10Timing']['Era'] = 'Phase2C1_timing'
upgradeProperties[2023]['2023D4Timing'] = deepcopy(upgradeProperties[2023]['2023D4'])
upgradeProperties[2023]['2023D4Timing']['Era'] = 'Phase2C2_timing'

#standard PU sequences
upgradeProperties[2023]['2023D7PU'] = deepcopy(upgradeProperties[2023]['2023D7'])
upgradeProperties[2023]['2023D7PU']['ScenToRun'] = ['GenSimHLBeamSpotFull','DigiFullTriggerPU','RecoFullGlobalPU', 'HARVESTFullGlobalPU']
upgradeProperties[2023]['2023D10PU'] = deepcopy(upgradeProperties[2023]['2023D10'])
upgradeProperties[2023]['2023D10PU']['ScenToRun'] = ['GenSimHLBeamSpotFull','DigiFullTriggerPU','RecoFullGlobalPU', 'HARVESTFullGlobalPU']
upgradeProperties[2023]['2023D4PU'] = deepcopy(upgradeProperties[2023]['2023D4'])
upgradeProperties[2023]['2023D4PU']['ScenToRun'] = ['GenSimHLBeamSpotFull','DigiFullTriggerPU','RecoFullGlobalPU', 'HARVESTFullGlobalPU']
upgradeProperties[2023]['2023D8PU'] = deepcopy(upgradeProperties[2023]['2023D8'])
upgradeProperties[2023]['2023D8PU']['ScenToRun'] = ['GenSimHLBeamSpotFull','DigiFullTriggerPU','RecoFullGlobalPU', 'HARVESTFullGlobalPU']
upgradeProperties[2023]['2023D9PU'] = deepcopy(upgradeProperties[2023]['2023D9'])
upgradeProperties[2023]['2023D9PU']['ScenToRun'] = ['GenSimHLBeamSpotFull','DigiFullTriggerPU','RecoFullGlobalPU', 'HARVESTFullGlobalPU']
upgradeProperties[2023]['2023D11PU'] = deepcopy(upgradeProperties[2023]['2023D11'])
upgradeProperties[2023]['2023D11PU']['ScenToRun'] = ['GenSimHLBeamSpotFull','DigiFullTriggerPU','RecoFullGlobalPU', 'HARVESTFullGlobalPU']


#Timing PU (for now copy ScenToRun of standard PU)
upgradeProperties[2023]['2023D7TimingPU'] = deepcopy(upgradeProperties[2023]['2023D7Timing'])
upgradeProperties[2023]['2023D7TimingPU']['ScenToRun'] = deepcopy(upgradeProperties[2023]['2023D7PU']['ScenToRun'])
upgradeProperties[2023]['2023D10TimingPU'] = deepcopy(upgradeProperties[2023]['2023D10Timing'])
upgradeProperties[2023]['2023D10TimingPU']['ScenToRun'] = deepcopy(upgradeProperties[2023]['2023D10PU']['ScenToRun'])
upgradeProperties[2023]['2023D4TimingPU'] = deepcopy(upgradeProperties[2023]['2023D4Timing'])
upgradeProperties[2023]['2023D4TimingPU']['ScenToRun'] = deepcopy(upgradeProperties[2023]['2023D4PU']['ScenToRun'])



from  Configuration.PyReleaseValidation.relval_steps import Kby

upgradeFragments=['FourMuPt_1_200_pythia8_cfi',
                  'SingleElectronPt10_pythia8_cfi',
                  'SingleElectronPt35_pythia8_cfi',
                  'SingleElectronPt1000_pythia8_cfi',
                  'SingleGammaPt10_pythia8_cfi',
                  'SingleGammaPt35_pythia8_cfi',
                  'SingleMuPt1_pythia8_cfi',
                  'SingleMuPt10_pythia8_cfi',
                  'SingleMuPt100_pythia8_cfi',
                  'SingleMuPt1000_pythia8_cfi',
                  'FourMuExtendedPt_1_200_pythia8_cfi',
                  'TenMuExtendedE_0_200_pythia8_cfi',
                  'DoubleElectronPt10Extended_pythia8_cfi',
                  'DoubleElectronPt35Extended_pythia8_cfi',
                  'DoubleElectronPt1000Extended_pythia8_cfi',
                  'DoubleGammaPt10Extended_pythia8_cfi',
                  'DoubleGammaPt35Extended_pythia8_cfi',
                  'DoubleMuPt1Extended_pythia8_cfi',
                  'DoubleMuPt10Extended_pythia8_cfi',
                  'DoubleMuPt100Extended_pythia8_cfi',
                  'DoubleMuPt1000Extended_pythia8_cfi',
                  'TenMuE_0_200_pythia8_cfi',
                  'SinglePiE50HCAL_pythia8_cfi',
                  'MinBias_13TeV_pythia8_TuneCUETP8M1_cfi', 
                  'TTbar_13TeV_TuneCUETP8M1_cfi',
                  'ZEE_13TeV_TuneCUETP8M1_cfi',
                  'QCD_Pt_600_800_13TeV_TuneCUETP8M1_cfi',
                  'Wjet_Pt_80_120_14TeV_TuneCUETP8M1_cfi',
                  'Wjet_Pt_3000_3500_14TeV_TuneCUETP8M1_cfi',
                  'LM1_sfts_14TeV_cfi',
                  'QCD_Pt_3000_3500_14TeV_TuneCUETP8M1_cfi',
                  'QCD_Pt_80_120_14TeV_TuneCUETP8M1_cfi',
                  'H200ChargedTaus_Tauola_14TeV_cfi',
                  'JpsiMM_14TeV_TuneCUETP8M1_cfi',
                  'TTbar_14TeV_TuneCUETP8M1_cfi',
                  'WE_14TeV_TuneCUETP8M1_cfi',
                  'ZTT_Tauola_All_hadronic_14TeV_TuneCUETP8M1_cfi',
                  'H130GGgluonfusion_14TeV_TuneCUETP8M1_cfi',
                  'PhotonJet_Pt_10_14TeV_TuneCUETP8M1_cfi',
                  'QQH1352T_Tauola_14TeV_TuneCUETP8M1_cfi',
                  'MinBias_14TeV_pythia8_TuneCUETP8M1_cfi',
                  'WM_14TeV_TuneCUETP8M1_cfi',
                  'ZMM_13TeV_TuneCUETP8M1_cfi',
                  'QCDForPF_14TeV_TuneCUETP8M1_cfi',
                  'DYToLL_M-50_14TeV_pythia8_cff',
                  'DYToTauTau_M-50_14TeV_pythia8_tauola_cff',
                  'ZEE_14TeV_TuneCUETP8M1_cfi',
                  'QCD_Pt_80_120_13TeV_TuneCUETP8M1_cfi',
                  'H125GGgluonfusion_13TeV_TuneCUETP8M1_cfi',
                  'QCD_Pt-20toInf_MuEnrichedPt15_TuneCUETP8M1_14TeV_pythia8_cff',
                  'ZMM_14TeV_TuneCUETP8M1_cfi',
                  'QCD_Pt-15To7000_TuneCUETP8M1_Flat_14TeV-pythia8_cff',
                  'H125GGgluonfusion_14TeV_TuneCUETP8M1_cfi',
                  'QCD_Pt_600_800_14TeV_TuneCUETP8M1_cfi',
                  'UndergroundCosmicSPLooseMu_cfi',
                  'BeamHalo_13TeV_cfi',
                  'H200ChargedTaus_Tauola_13TeV_cfi',
                  'ADDMonoJet_13TeV_d3MD3_TuneCUETP8M1_cfi',
                  'ZpMM_13TeV_TuneCUETP8M1_cfi',
                  'QCD_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi',
                  'WpM_13TeV_TuneCUETP8M1_cfi',
                  'SingleNuE10_cfi.py',
                  'TTbarLepton_13TeV_TuneCUETP8M1_cfi',
                  'WE_13TeV_TuneCUETP8M1_cfi',
                  'WM_13TeV_TuneCUETP8M1_cfi',
                  'ZTT_All_hadronic_13TeV_TuneCUETP8M1_cfi',
                  'PhotonJet_Pt_10_13TeV_TuneCUETP8M1_cfi',
                  'QQH1352T_13TeV_TuneCUETP8M1_cfi',
                  'Wjet_Pt_80_120_13TeV_TuneCUETP8M1_cfi',
                  'Wjet_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi',
                  'SMS-T1tttt_mGl-1500_mLSP-100_13TeV-pythia8_cfi',
                  'QCDForPF_13TeV_TuneCUETP8M1_cfi',
                  'PYTHIA8_PhiToMuMu_TuneCUETP8M1_13TeV_cff',
                  'RSKKGluon_m3000GeV_13TeV_TuneCUETP8M1_cff',
                  'ZpMM_2250_13TeV_TuneCUETP8M1_cfi',
                  'ZpEE_2250_13TeV_TuneCUETP8M1_cfi',
                  'ZpTT_1500_13TeV_TuneCUETP8M1_cfi',
                  'Upsilon1SToMuMu_forSTEAM_13TeV_TuneCUETP8M1_cfi',
                  'EtaBToJpsiJpsi_forSTEAM_TuneCUEP8M1_13TeV_cfi',
                  'JpsiMuMu_Pt-8_forSTEAM_13TeV_TuneCUETP8M1_cfi',
                  'BuMixing_BMuonFilter_forSTEAM_13TeV_TuneCUETP8M1_cfi',
                  'HSCPstop_M_200_TuneCUETP8M1_13TeV_pythia8_cff',
                  'RSGravitonToGammaGamma_kMpl01_M_3000_TuneCUETP8M1_13TeV_pythia8_cfi',
                  'WprimeToENu_M-2000_TuneCUETP8M1_13TeV-pythia8_cff',
                  'DisplacedSUSY_stopToBottom_M_300_1000mm_TuneCUETP8M1_13TeV_pythia8_cff',
]

howMuches={'FourMuPt_1_200_pythia8_cfi':Kby(10,100),
           'TenMuE_0_200_pythia8_cfi':Kby(10,100),
           'FourMuExtendedPt_1_200_pythia8_cfi':Kby(10,100),
           'TenMuExtendedE_0_200_pythia8_cfi':Kby(10,100),
           'SingleElectronPt10_pythia8_cfi':Kby(9,100),
           'SingleElectronPt35_pythia8_cfi':Kby(9,100),
           'SingleElectronPt1000_pythia8_cfi':Kby(9,50),
           'SingleGammaPt10_pythia8_cfi':Kby(9,100),
           'SingleGammaPt35_pythia8_cfi':Kby(9,50),
           'SingleMuPt1_pythia8_cfi':Kby(25,100),
           'SingleMuPt10_pythia8_cfi':Kby(25,100),
           'SingleMuPt100_pythia8_cfi':Kby(9,100),
           'SingleMuPt1000_pythia8_cfi':Kby(9,100),
           'DoubleElectronPt10Extended_pythia8_cfi':Kby(9,100),
           'DoubleElectronPt35Extended_pythia8_cfi':Kby(9,100),
           'DoubleElectronPt1000Extended_pythia8_cfi':Kby(9,50),
           'DoubleGammaPt10Extended_pythia8_cfi':Kby(9,100),
           'DoubleGammaPt35Extended_pythia8_cfi':Kby(9,50),
           'DoubleMuPt1Extended_pythia8_cfi':Kby(25,100),
           'DoubleMuPt10Extended_pythia8_cfi':Kby(25,100),
           'DoubleMuPt100Extended_pythia8_cfi':Kby(9,100),
           'DoubleMuPt1000Extended_pythia8_cfi':Kby(9,100),
           'SinglePiE50HCAL_pythia8_cfi':Kby(10,100),
           'QCD_Pt_600_800_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'Wjet_Pt_80_120_14TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'Wjet_Pt_3000_3500_14TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'LM1_sfts_14TeV_cfi':Kby(9,100),
           'QCD_Pt_3000_3500_14TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'QCD_Pt_80_120_14TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'H200ChargedTaus_Tauola_14TeV_cfi':Kby(9,100),
           'JpsiMM_14TeV_TuneCUETP8M1_cfi':Kby(66,100),
           'TTbar_14TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'WE_14TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'ZEE_13TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'ZTT_Tauola_All_hadronic_14TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'H130GGgluonfusion_14TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'PhotonJet_Pt_10_14TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'QQH1352T_Tauola_14TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'MinBias_14TeV_pythia8_TuneCUETP8M1_cfi':Kby(90,100),
           'WM_14TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'ZMM_13TeV_TuneCUETP8M1_cfi':Kby(18,100),
           'QCDForPF_14TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'DYToLL_M-50_14TeV_pythia8_cff':Kby(9,100),
           'DYToTauTau_M-50_14TeV_pythia8_tauola_cff':Kby(9,100),
           'TTbar_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'MinBias_13TeV_pythia8_TuneCUETP8M1_cfi':Kby(90,100),
           'ZEE_14TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'QCD_Pt_80_120_13TeV_TuneCUETP8M1_cfi':Kby(9,100),
           'H125GGgluonfusion_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'QCD_Pt-20toInf_MuEnrichedPt15_TuneCUETP8M1_14TeV_pythia8_cff':Kby(9,100),
           'ZMM_14TeV_TuneCUETP8M1_cfi':Kby(18,100),
           'QCD_Pt-15To7000_TuneCUETP8M1_Flat_14TeV-pythia8_cff':Kby(9,50),
           'H125GGgluonfusion_14TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'QCD_Pt_600_800_14TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'UndergroundCosmicSPLooseMu_cfi':Kby(9,50),
           'BeamHalo_13TeV_cfi':Kby(9,50),
           'H200ChargedTaus_Tauola_13TeV_cfi':Kby(9,50),
           'ADDMonoJet_13TeV_d3MD3_TuneCUETP8M1_cfi':Kby(9,50),
           'ZpMM_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'QCD_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'WpM_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'SingleNuE10_cfi.py':Kby(9,50),
           'TTbarLepton_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'WE_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'WM_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'ZTT_All_hadronic_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'PhotonJet_Pt_10_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'QQH1352T_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'Wjet_Pt_80_120_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'Wjet_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'SMS-T1tttt_mGl-1500_mLSP-100_13TeV-pythia8_cfi':Kby(9,50),
           'QCDForPF_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'PYTHIA8_PhiToMuMu_TuneCUETP8M1_13TeV_cff':Kby(9,50),
           'RSKKGluon_m3000GeV_13TeV_TuneCUETP8M1_cff':Kby(9,50),
           'ZpMM_2250_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'ZpEE_2250_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'ZpTT_1500_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'Upsilon1SToMuMu_forSTEAM_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'EtaBToJpsiJpsi_forSTEAM_TuneCUEP8M1_13TeV_cfi':Kby(9,50),
           'JpsiMuMu_Pt-8_forSTEAM_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'BuMixing_BMuonFilter_forSTEAM_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'HSCPstop_M_200_TuneCUETP8M1_13TeV_pythia8_cff':Kby(9,50),
           'RSGravitonToGammaGamma_kMpl01_M_3000_TuneCUETP8M1_13TeV_pythia8_cfi':Kby(9,50),
           'WprimeToENu_M-2000_TuneCUETP8M1_13TeV-pythia8_cff':Kby(9,50),
           'DisplacedSUSY_stopToBottom_M_300_1000mm_TuneCUETP8M1_13TeV_pythia8_cff':Kby(9,50),
}

upgradeDatasetFromFragment={'FourMuPt_1_200_pythia8_cfi': 'FourMuPt1_200',
                            'FourMuExtendedPt_1_200_pythia8_cfi': 'FourMuExtendedPt1_200',
                            'TenMuE_0_200_pythia8_cfi': 'TenMuE_0_200',
                            'TenMuExtendedE_0_200_pythia8_cfi': 'TenMuExtendedE_0_200',
                            'SingleElectronPt10_pythia8_cfi' : 'SingleElectronPt10',
                            'SingleElectronPt35_pythia8_cfi' : 'SingleElectronPt35',
                            'SingleElectronPt1000_pythia8_cfi' : 'SingleElectronPt1000',
                            'SingleGammaPt10_pythia8_cfi' : 'SingleGammaPt10',
                            'SingleGammaPt35_pythia8_cfi' : 'SingleGammaPt35',
                            'SingleMuPt1_pythia8_cfi' : 'SingleMuPt1',
                            'SingleMuPt10_pythia8_cfi' : 'SingleMuPt10',
                            'SingleMuPt100_pythia8_cfi' : 'SingleMuPt100',
                            'SingleMuPt1000_pythia8_cfi' : 'SingleMuPt1000',
                            'DoubleElectronPt10Extended_pythia8_cfi' : 'SingleElectronPt10Extended',
                            'DoubleElectronPt35Extended_pythia8_cfi' : 'SingleElectronPt35Extended',
                            'DoubleElectronPt1000Extended_pythia8_cfi' : 'SingleElectronPt1000Extended',
                            'DoubleGammaPt10Extended_pythia8_cfi' : 'SingleGammaPt10Extended',
                            'DoubleGammaPt35Extended_pythia8_cfi' : 'SingleGammaPt35Extended',
                            'DoubleMuPt1Extended_pythia8_cfi' : 'SingleMuPt1Extended',
                            'DoubleMuPt10Extended_pythia8_cfi' : 'SingleMuPt10Extended',
                            'DoubleMuPt100Extended_pythia8_cfi' : 'SingleMuPt100Extended',
                            'DoubleMuPt1000Extended_pythia8_cfi' : 'SingleMuPt1000Extended',
                            'SinglePiE50HCAL_pythia8_cfi' : 'SinglePiE50HCAL',
                            'QCD_Pt_600_800_13TeV_TuneCUETP8M1_cfi' : 'QCD_Pt_600_800_13',
                            'Wjet_Pt_80_120_14TeV_TuneCUETP8M1_cfi' : 'Wjet_Pt_80_120_14TeV',
                            'Wjet_Pt_3000_3500_14TeV_TuneCUETP8M1_cfi' : 'Wjet_Pt_3000_3500_14TeV',
                            'LM1_sfts_14TeV_cfi' : 'LM1_sfts_14TeV',
                            'QCD_Pt_3000_3500_14TeV_TuneCUETP8M1_cfi' : 'QCD_Pt_3000_3500_14TeV',
                            'QCD_Pt_80_120_14TeV_TuneCUETP8M1_cfi' : 'QCD_Pt_80_120_14TeV',
                            'H200ChargedTaus_Tauola_14TeV_cfi' : 'Higgs200ChargedTaus_14TeV',
                            'JpsiMM_14TeV_TuneCUETP8M1_cfi' : 'JpsiMM_14TeV',
                            'TTbar_14TeV_TuneCUETP8M1_cfi' : 'TTbar_14TeV',
                            'WE_14TeV_TuneCUETP8M1_cfi' : 'WE_14TeV',
                            'ZEE_13TeV_TuneCUETP8M1_cfi' : 'ZEE_13',
                            'ZTT_Tauola_All_hadronic_14TeV_TuneCUETP8M1_cfi' : 'ZTT_14TeV',
                            'H130GGgluonfusion_14TeV_TuneCUETP8M1_cfi' : 'H130GGgluonfusion_14TeV',
                            'PhotonJet_Pt_10_14TeV_TuneCUETP8M1_cfi' : 'PhotonJets_Pt_10_14TeV',
                            'QQH1352T_Tauola_14TeV_TuneCUETP8M1_cfi' : 'QQH1352T_Tauola_14TeV',
                            'MinBias_14TeV_pythia8_TuneCUETP8M1_cfi' : 'MinBias_14TeV',
                            'WM_14TeV_TuneCUETP8M1_cfi' : 'WM_14TeV',
                            'ZMM_13TeV_TuneCUETP8M1_cfi' : 'ZMM_13',
                            'QCDForPF_14TeV_TuneCUETP8M1_cfi' : 'QCDForPF_14TeV',
                            'DYToLL_M-50_14TeV_pythia8_cff' : 'DYToLL_M_50_14TeV',
                            'DYToTauTau_M-50_14TeV_pythia8_tauola_cff' : 'DYtoTauTau_M_50_14TeV',
                            'TTbar_13TeV_TuneCUETP8M1_cfi' : 'TTbar_13',
                            'MinBias_13TeV_pythia8_TuneCUETP8M1_cfi' : 'MinBias_13',
                            'ZEE_14TeV_TuneCUETP8M1_cfi' : 'ZEE_14',
                            'QCD_Pt_80_120_13TeV_TuneCUETP8M1_cfi' : 'QCD_Pt_80_120_13',
                            'H125GGgluonfusion_13TeV_TuneCUETP8M1_cfi' : 'H125GGgluonfusion_13',
                            'QCD_Pt-20toInf_MuEnrichedPt15_TuneCUETP8M1_14TeV_pythia8_cff' : 'QCD_Pt-20toInf_MuEnrichedPt15_14TeV',
                            'ZMM_14TeV_TuneCUETP8M1_cfi' : 'ZMM_14',
                            'QCD_Pt-15To7000_TuneCUETP8M1_Flat_14TeV-pythia8_cff' : 'QCD_Pt-15To7000_Flat_14TeV',
                            'H125GGgluonfusion_14TeV_TuneCUETP8M1_cfi' : 'H125GGgluonfusion_14',
                            'QCD_Pt_600_800_14TeV_TuneCUETP8M1_cfi' : 'QCD_Pt_600_800_14',
                            'UndergroundCosmicSPLooseMu_cfi': 'CosmicsSPLoose_UP15',
                            'BeamHalo_13TeV_cfi': 'BeamHalo_13',
                            'H200ChargedTaus_Tauola_13TeV_cfi': 'Higgs200ChargedTaus_13',
                            'ADDMonoJet_13TeV_d3MD3_TuneCUETP8M1_cfi': 'ADDMonoJet_d3MD3_13',
                            'ZpMM_13TeV_TuneCUETP8M1_cfi': 'ZpMM_13',
                            'QCD_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi': 'QCD_Pt_3000_3500_13',
                            'WpM_13TeV_TuneCUETP8M1_cfi': 'WpM_13',
                            'SingleNuE10_cfi.py': 'NuGun_UP15',
                            'TTbarLepton_13TeV_TuneCUETP8M1_cfi': 'TTbarLepton_13',
                            'WE_13TeV_TuneCUETP8M1_cfi': 'WE_13',
                            'WM_13TeV_TuneCUETP8M1_cfi': 'WM_13',
                            'ZTT_All_hadronic_13TeV_TuneCUETP8M1_cfi': 'ZTT_13',
                            'PhotonJet_Pt_10_13TeV_TuneCUETP8M1_cfi': 'PhotonJets_Pt_10_13',
                            'QQH1352T_13TeV_TuneCUETP8M1_cfi': 'QQH1352T_13',
                            'Wjet_Pt_80_120_13TeV_TuneCUETP8M1_cfi': 'Wjet_Pt_80_120_13',
                            'Wjet_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi': 'Wjet_Pt_3000_3500_13',
                            'SMS-T1tttt_mGl-1500_mLSP-100_13TeV-pythia8_cfi': 'SMS-T1tttt_mGl-1500_mLSP-100_13',
                            'QCDForPF_13TeV_TuneCUETP8M1_cfi': 'QCD_FlatPt_15_3000HS_13',
                            'PYTHIA8_PhiToMuMu_TuneCUETP8M1_13TeV_cff': 'PhiToMuMu_13',
                            'RSKKGluon_m3000GeV_13TeV_TuneCUETP8M1_cff': 'RSKKGluon_m3000GeV_13',
                            'ZpMM_2250_13TeV_TuneCUETP8M1_cfi': 'ZpMM_2250_13',
                            'ZpEE_2250_13TeV_TuneCUETP8M1_cfi': 'ZpEE_2250_13',
                            'ZpTT_1500_13TeV_TuneCUETP8M1_cfi': 'ZpTT_1500_13',
                            'Upsilon1SToMuMu_forSTEAM_13TeV_TuneCUETP8M1_cfi': 'Upsilon1SToMuMu_13',
                            'EtaBToJpsiJpsi_forSTEAM_TuneCUEP8M1_13TeV_cfi': 'EtaBToJpsiJpsi_13',
                            'JpsiMuMu_Pt-8_forSTEAM_13TeV_TuneCUETP8M1_cfi': 'JpsiMuMu_Pt-8',
                            'BuMixing_BMuonFilter_forSTEAM_13TeV_TuneCUETP8M1_cfi': 'BuMixing_13',
                            'HSCPstop_M_200_TuneCUETP8M1_13TeV_pythia8_cff': 'HSCPstop_M_200_13',
                            'RSGravitonToGammaGamma_kMpl01_M_3000_TuneCUETP8M1_13TeV_pythia8_cfi': 'RSGravitonToGaGa_13',
                            'WprimeToENu_M-2000_TuneCUETP8M1_13TeV-pythia8_cff': 'WpToENu_M-2000_13',
                            'DisplacedSUSY_stopToBottom_M_300_1000mm_TuneCUETP8M1_13TeV_pythia8_cff': 'DisplacedSUSY_stopToBottom_M_300_1000mm_13',
}
