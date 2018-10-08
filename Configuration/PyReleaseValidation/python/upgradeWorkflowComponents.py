from copy import deepcopy

# DON'T CHANGE THE ORDER, only append new keys. Otherwise the numbering for the runTheMatrix tests will change.

upgradeKeys = {}

upgradeKeys[2017] = [
    '2017',
    '2017PU',
    '2017Design',
    '2017DesignPU',
    '2018',
    '2018PU',
    '2018Design',
    '2018DesignPU',
    '2019',
#    '2019PU',
    '2019Design',
#    '2019DesignPU',
]

upgradeKeys[2023] = [
    '2023D17',
    '2023D17PU',
    '2023D19',
    '2023D19PU',
    '2023D21',
    '2023D21PU',
    '2023D22',
    '2023D22PU',
    '2023D23',
    '2023D23PU',
    '2023D24',
    '2023D24PU',
    '2023D25',
    '2023D25PU',
    '2023D26',
    '2023D26PU',
    '2023D27',
    '2023D27PU',
    '2023D28',
    '2023D28PU',
    '2023D29',
    '2023D29PU',
    '2023D30',
    '2023D30PU',
    '2023D31',
    '2023D31PU',
]

# pre-generation of WF numbers
numWFStart={
    2017: 10000,
    2023: 20000,
}
numWFSkip=200
# temporary measure to keep other WF numbers the same
numWFConflict = [[11800,12000],[12200,12400],[20800,21200],[25000,26000],[50000,51000]]
numWFAll={
    2017: [],
    2023: []
}

for year in upgradeKeys:
    for i in range(0,len(upgradeKeys[year])):
        numWFtmp = numWFStart[year] if i==0 else (numWFAll[year][i-1] + numWFSkip)
        for conflict in numWFConflict:
            if numWFtmp>=conflict[0] and numWFtmp<conflict[1]:
                numWFtmp = conflict[1]
                break
        numWFAll[year].append(numWFtmp)

# steps for baseline and for variations
upgradeSteps={}
upgradeSteps['baseline'] = {
    'steps' : [
        'GenSimFull',
        'GenSimHLBeamSpotFull',
        'GenSimHLBeamSpotFull14',
        'DigiFull',
        'DigiFullTrigger',
        'RecoFullLocal',
        'RecoFull',
        'RecoFullGlobal',
        'HARVESTFull',
        'FastSim',
        'HARVESTFast',
        'HARVESTFullGlobal',
        'ALCAFull',
        'NanoFull',
    ],
    'PU' : [
        'DigiFullTrigger',
        'RecoFullLocal',
        'RecoFullGlobal',
        'DigiFull',
        'RecoFull',
        'HARVESTFull',
        'HARVESTFullGlobal',
    ],
    'suffix' : '',
    'offset' : 0.0,
}
upgradeSteps['trackingOnly'] = {
    'steps' : [
        'RecoFull',
        'HARVESTFull',
        'RecoFullGlobal',
        'HARVESTFullGlobal',
    ],
    'PU' : [],
    'suffix' : '_trackingOnly',
    'offset' : 0.1,
}
upgradeSteps['trackingRun2'] = {
    'steps' : [
        'RecoFull',
    ],
    'PU' : [],
    'suffix' : '_trackingRun2',
    'offset' : 0.2,
}
upgradeSteps['trackingOnlyRun2'] = {
    'steps' : [
        'RecoFull',
        'HARVESTFull',
    ],
    'PU' : [],
    'suffix' : '_trackingOnlyRun2',
    'offset' : 0.3,
}
upgradeSteps['trackingLowPU'] = {
    'steps' : [
        'RecoFull',
    ],
    'PU' : [],
    'suffix' : '_trackingLowPU',
    'offset' : 0.4,
}
upgradeSteps['pixelTrackingOnly'] = {
    'steps' : [
        'RecoFull',
        'HARVESTFull',
        'RecoFullGlobal',
        'HARVESTFullGlobal',
    ],
    'PU' : [],
    'suffix' : '_pixelTrackingOnly',
    'offset' : 0.5,
}
upgradeSteps['Timing'] = {
    'steps' : upgradeSteps['baseline']['steps'],
    'PU' : upgradeSteps['baseline']['PU'],
    'suffix' : '_Timing',
    'offset' : 0.11,
}
upgradeSteps['Neutron'] = {
    'steps' : [
        'GenSimFull',
        'GenSimHLBeamSpotFull',
        'GenSimHLBeamSpotFull14',
        'DigiFull',
        'DigiFullTrigger',
    ],
    'PU' : [
        'DigiFull',
        'DigiFullTrigger',
    ],
    'suffix' : '_Neutron',
    'offset' : 0.12,
}
upgradeSteps['heCollapse'] = {
    'steps' : [
        'GenSimFull',
        'DigiFull',
        'RecoFull',
        'HARVESTFull',
        'ALCAFull',
    ],
    'PU' : [
        'DigiFull',
        'RecoFull',
        'HARVESTFull',
    ],
    'suffix' : '_heCollapse',
    'offset' : 0.6,
}
upgradeSteps['Premix'] = {
    'steps' : [],
    'PU': [
        'PremixFull',
        'PremixHLBeamSpotFull',
        'PremixHLBeamSpotFull14',
    ],
    'suffix': '_Premix',
    'offset': 0.97,
}
# Premix stage2 is derived from baseline+PU in relval_upgrade.py
premixS2_offset = 0.98
# Premix combined stage1+stage2 is derived for Premix+PU and baseline+PU in relval_upgrade.py
premixS1S2_offset = 0.99

upgradeProperties = {}

upgradeProperties[2017] = {
    '2017' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2017_realistic',
        'HLTmenu': '@relval2017',
        'Era' : 'Run2_2017',
        'ScenToRun' : ['GenSimFull','DigiFull','RecoFull','HARVESTFull','ALCAFull'],
    },
    '2017Design' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2017_design',
        'HLTmenu': '@relval2017',
        'Era' : 'Run2_2017',
        'BeamSpot': 'GaussSigmaZ4cm',
        'ScenToRun' : ['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
    },
    '2018' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2018_realistic',
        'HLTmenu': '@relval2018',
        'Era' : 'Run2_2018',
        'BeamSpot': 'Realistic25ns13TeVEarly2018Collision',
        'ScenToRun' : ['GenSimFull','DigiFull','RecoFull','HARVESTFull','ALCAFull','NanoFull'],
    },
    '2018Design' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2018_design',
        'HLTmenu': '@relval2018',
        'Era' : 'Run2_2018',
        'BeamSpot': 'GaussSigmaZ4cm',
        'ScenToRun' : ['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
    },
    '2019' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2019_realistic',
        'HLTmenu': '@relval2017',
        'Era' : 'Run3',
        'ScenToRun' : ['GenSimFull','DigiFull','RecoFull','HARVESTFull','ALCAFull'],
    },
    '2019Design' : {
        'Geom' : 'DB:Extended',
        'GT' : 'auto:phase1_2019_design',
        'HLTmenu': '@relval2017',
        'Era' : 'Run3',
        'BeamSpot': 'GaussSigmaZ4cm',
        'ScenToRun' : ['GenSimFull','DigiFull','RecoFull','HARVESTFull'],
    },
}

upgradeProperties[2017]['2017PU'] = deepcopy(upgradeProperties[2017]['2017'])
upgradeProperties[2017]['2017PU']['ScenToRun'] = ['GenSimFull','DigiFullPU','RecoFullPU','HARVESTFullPU']
upgradeProperties[2017]['2017DesignPU'] = deepcopy(upgradeProperties[2017]['2017Design'])
upgradeProperties[2017]['2017DesignPU']['ScenToRun'] = ['GenSimFull','DigiFullPU','RecoFullPU','HARVESTFullPU']
upgradeProperties[2017]['2018PU'] = deepcopy(upgradeProperties[2017]['2018'])
upgradeProperties[2017]['2018PU']['ScenToRun'] = ['GenSimFull','DigiFullPU','RecoFullPU','HARVESTFullPU','NanoFull']
upgradeProperties[2017]['2018DesignPU'] = deepcopy(upgradeProperties[2017]['2018Design'])
upgradeProperties[2017]['2018DesignPU']['ScenToRun'] = ['GenSimFull','DigiFullPU','RecoFullPU','HARVESTFullPU']

upgradeProperties[2023] = {
    '2023D17' : {
        'Geom' : 'Extended2023D17',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic',
        'Era' : 'Phase2',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
     },
    '2023D19' : {
        'Geom' : 'Extended2023D19',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic',
        'Era' : 'Phase2_timing_layer',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
    },
    '2023D21' : {
        'Geom' : 'Extended2023D21',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic',
        'Era' : 'Phase2',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
     },
    '2023D22' : {
        'Geom' : 'Extended2023D22',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic',
        'Era' : 'Phase2',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
     },
    '2023D23' : {
        'Geom' : 'Extended2023D23',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic',
        'Era' : 'Phase2',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
     },
    '2023D24' : {
        'Geom' : 'Extended2023D24',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic',
        'Era' : 'Phase2_timing_layer_new',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
    },
    '2023D25' : {
        'Geom' : 'Extended2023D25',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic',
        'Era' : 'Phase2_timing_layer_new',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
    },
    '2023D26' : {
        'Geom' : 'Extended2023D26',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic',
        'Era' : 'Phase2',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
     },
    '2023D27' : {
        'Geom' : 'Extended2023D27',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic',
        'Era' : 'Phase2',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
     },
    '2023D28' : {
        'Geom' : 'Extended2023D28',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic',
        'Era' : 'Phase2C4',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
     },
    '2023D29' : {
        'Geom' : 'Extended2023D29',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic',
        'Era' : 'Phase2',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
     },
    '2023D30' : {
        'Geom' : 'Extended2023D30',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic',
        'Era' : 'Phase2C4',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
     },
    '2023D31' : {
        'Geom' : 'Extended2023D31',
        'HLTmenu': '@fake2',
        'GT' : 'auto:phase2_realistic',
        'Era' : 'Phase2C6',
        'ScenToRun' : ['GenSimHLBeamSpotFull','DigiFullTrigger','RecoFullGlobal', 'HARVESTFullGlobal'],
     },
}



#standard PU sequences
upgradeProperties[2023]['2023D17PU'] = deepcopy(upgradeProperties[2023]['2023D17'])
upgradeProperties[2023]['2023D17PU']['ScenToRun'] = ['GenSimHLBeamSpotFull','DigiFullTriggerPU','RecoFullGlobalPU', 'HARVESTFullGlobalPU']
upgradeProperties[2023]['2023D19PU'] = deepcopy(upgradeProperties[2023]['2023D19'])
upgradeProperties[2023]['2023D19PU']['ScenToRun'] = ['GenSimHLBeamSpotFull','DigiFullTriggerPU','RecoFullGlobalPU', 'HARVESTFullGlobalPU']
upgradeProperties[2023]['2023D21PU'] = deepcopy(upgradeProperties[2023]['2023D21'])
upgradeProperties[2023]['2023D21PU']['ScenToRun'] = ['GenSimHLBeamSpotFull','DigiFullTriggerPU','RecoFullGlobalPU', 'HARVESTFullGlobalPU']
upgradeProperties[2023]['2023D22PU'] = deepcopy(upgradeProperties[2023]['2023D22'])
upgradeProperties[2023]['2023D22PU']['ScenToRun'] = ['GenSimHLBeamSpotFull','DigiFullTriggerPU','RecoFullGlobalPU', 'HARVESTFullGlobalPU']
upgradeProperties[2023]['2023D23PU'] = deepcopy(upgradeProperties[2023]['2023D23'])
upgradeProperties[2023]['2023D23PU']['ScenToRun'] = ['GenSimHLBeamSpotFull','DigiFullTriggerPU','RecoFullGlobalPU', 'HARVESTFullGlobalPU']
upgradeProperties[2023]['2023D24PU'] = deepcopy(upgradeProperties[2023]['2023D24'])
upgradeProperties[2023]['2023D24PU']['ScenToRun'] = ['GenSimHLBeamSpotFull','DigiFullTriggerPU','RecoFullGlobalPU', 'HARVESTFullGlobalPU']
upgradeProperties[2023]['2023D25PU'] = deepcopy(upgradeProperties[2023]['2023D25'])
upgradeProperties[2023]['2023D25PU']['ScenToRun'] = ['GenSimHLBeamSpotFull','DigiFullTriggerPU','RecoFullGlobalPU', 'HARVESTFullGlobalPU']
upgradeProperties[2023]['2023D26PU'] = deepcopy(upgradeProperties[2023]['2023D26'])
upgradeProperties[2023]['2023D26PU']['ScenToRun'] = ['GenSimHLBeamSpotFull','DigiFullTriggerPU','RecoFullGlobalPU', 'HARVESTFullGlobalPU']
upgradeProperties[2023]['2023D27PU'] = deepcopy(upgradeProperties[2023]['2023D27'])
upgradeProperties[2023]['2023D27PU']['ScenToRun'] = ['GenSimHLBeamSpotFull','DigiFullTriggerPU','RecoFullGlobalPU', 'HARVESTFullGlobalPU']
upgradeProperties[2023]['2023D28PU'] = deepcopy(upgradeProperties[2023]['2023D28'])
upgradeProperties[2023]['2023D28PU']['ScenToRun'] = ['GenSimHLBeamSpotFull','DigiFullTriggerPU','RecoFullGlobalPU', 'HARVESTFullGlobalPU']
upgradeProperties[2023]['2023D29PU'] = deepcopy(upgradeProperties[2023]['2023D29'])
upgradeProperties[2023]['2023D29PU']['ScenToRun'] = ['GenSimHLBeamSpotFull','DigiFullTriggerPU','RecoFullGlobalPU', 'HARVESTFullGlobalPU']
upgradeProperties[2023]['2023D30PU'] = deepcopy(upgradeProperties[2023]['2023D30'])
upgradeProperties[2023]['2023D30PU']['ScenToRun'] = ['GenSimHLBeamSpotFull','DigiFullTriggerPU','RecoFullGlobalPU', 'HARVESTFullGlobalPU']
upgradeProperties[2023]['2023D31PU'] = deepcopy(upgradeProperties[2023]['2023D31'])
upgradeProperties[2023]['2023D31PU']['ScenToRun'] = ['GenSimHLBeamSpotFull','DigiFullTriggerPU','RecoFullGlobalPU', 'HARVESTFullGlobalPU']


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
                  'TenE_E_0_200_pythia8_cfi',
                  'FlatRandomPtAndDxyGunProducer_cfi',
                  'TenTau_E_15_500_pythia8_cfi',
                  'SinglePiPt25Eta1p7_2p7_cfi',
                  'SingleMuPt15Eta1p7_2p7_cfi',
                  'SingleGammaPt25Eta1p7_2p7_cfi',
                  'SingleElectronPt15Eta1p7_2p7_cfi',
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
           'SinglePiE50HCAL_pythia8_cfi':Kby(50,500),
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
           'QCDForPF_13TeV_TuneCUETP8M1_cfi':Kby(50,100),
           'PYTHIA8_PhiToMuMu_TuneCUETP8M1_13TeV_cff':Kby(9,50),
           'RSKKGluon_m3000GeV_13TeV_TuneCUETP8M1_cff':Kby(9,50),
           'ZpMM_2250_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'ZpEE_2250_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'ZpTT_1500_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'Upsilon1SToMuMu_forSTEAM_13TeV_TuneCUETP8M1_cfi':Kby(9,50),
           'EtaBToJpsiJpsi_forSTEAM_TuneCUEP8M1_13TeV_cfi':Kby(9,50),
           'JpsiMuMu_Pt-8_forSTEAM_13TeV_TuneCUETP8M1_cfi':Kby(3100,100000),
           'BuMixing_BMuonFilter_forSTEAM_13TeV_TuneCUETP8M1_cfi':Kby(900,10000),
           'HSCPstop_M_200_TuneCUETP8M1_13TeV_pythia8_cff':Kby(9,50),
           'RSGravitonToGammaGamma_kMpl01_M_3000_TuneCUETP8M1_13TeV_pythia8_cfi':Kby(9,50),
           'WprimeToENu_M-2000_TuneCUETP8M1_13TeV-pythia8_cff':Kby(9,50),
           'DisplacedSUSY_stopToBottom_M_300_1000mm_TuneCUETP8M1_13TeV_pythia8_cff':Kby(9,50),
           'TenE_E_0_200_pythia8_cfi':Kby(9,100),
           'FlatRandomPtAndDxyGunProducer_cfi':Kby(9,100),
           'TenTau_E_15_500_pythia8_cfi':Kby(9,100),
           'SinglePiPt25Eta1p7_2p7_cfi':Kby(9,100),
           'SingleMuPt15Eta1p7_2p7_cfi':Kby(9,100),
           'SingleGammaPt25Eta1p7_2p7_cfi':Kby(9,100),
           'SingleElectronPt15Eta1p7_2p7_cfi':Kby(9,100),
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
                            'UndergroundCosmicSPLooseMu_cfi': 'CosmicsSPLoose',
                            'BeamHalo_13TeV_cfi': 'BeamHalo_13',
                            'H200ChargedTaus_Tauola_13TeV_cfi': 'Higgs200ChargedTaus_13',
                            'ADDMonoJet_13TeV_d3MD3_TuneCUETP8M1_cfi': 'ADDMonoJet_d3MD3_13',
                            'ZpMM_13TeV_TuneCUETP8M1_cfi': 'ZpMM_13',
                            'QCD_Pt_3000_3500_13TeV_TuneCUETP8M1_cfi': 'QCD_Pt_3000_3500_13',
                            'WpM_13TeV_TuneCUETP8M1_cfi': 'WpM_13',
                            'SingleNuE10_cfi.py': 'NuGun',
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
                            'TenE_E_0_200_pythia8_cfi': 'TenE_0_200',
                            'FlatRandomPtAndDxyGunProducer_cfi': 'DisplacedMuonsDxy_0_500',
                            'TenTau_E_15_500_pythia8_cfi':'TenTau_15_500',
                            'SinglePiPt25Eta1p7_2p7_cfi':'SinglePiPt25Eta1p7_2p7',
                            'SingleMuPt15Eta1p7_2p7_cfi':'SingleMuPt15Eta1p7_2p7',
                            'SingleGammaPt25Eta1p7_2p7_cfi':'SingleGammaPt25Eta1p7_2p7',
                            'SingleElectronPt15Eta1p7_2p7_cfi':'SingleElectronPt15Eta1p7_2p7',
}
