
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = {}

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used


#part gun
workflows[3100] = ['', ['FourMuPt1_200_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3101] = ['', ['SingleElectronPt10_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3102] = ['', ['SingleElectronPt1000_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3103] = ['', ['SingleElectronPt35_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3104] = ['', ['SingleGammaPt10_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3105] = ['', ['SingleGammaPt35_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3106] = ['', ['SingleMuPt1_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3107] = ['', ['SingleMuPt10_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3108] = ['', ['SingleMuPt100_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3109] = ['', ['SingleMuPt1000_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]

#std wf @8TeV
workflows[3110] = ['', ['TTbarLepton_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3111] = ['', ['Wjet_Pt_80_120_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3112] = ['', ['Wjet_Pt_3000_3500_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3113] = ['', ['LM1_sfts_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]

workflows[3114] = ['', ['QCD_Pt_3000_3500_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3115] = ['', ['QCD_Pt_600_800_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3116] = ['', ['QCD_Pt_80_120_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]

workflows[3117] = ['', ['Higgs200ChargedTaus_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3118] = ['', ['JpsiMM_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3119] = ['', ['TTbar_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3120] = ['', ['WE_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3121] = ['', ['ZEE_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3122] = ['', ['ZTT_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3123] = ['', ['H130GGgluonfusion_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3124] = ['', ['PhotonJets_Pt_10_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3125] = ['', ['QQH1352T_Tauola_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]

workflows[3126] = ['', ['MinBias_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3127] = ['', ['WM_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3128] = ['', ['ZMM_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]

workflows[3129] = ['', ['ADDMonoJet_d3MD3_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3130] = ['', ['ZpMM_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]
workflows[3131] = ['', ['WpM_UPGPhase1','DIGIUP','RECOUP','HARVESTUP']]


#std wf @14TeV
workflows[3110] = ['', ['TTbarLepton_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3111] = ['', ['Wjet_Pt_80_120_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3112] = ['', ['Wjet_Pt_3000_3500_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3113] = ['', ['LM1_sfts_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]


workflows[3114] = ['', ['QCD_Pt_3000_3500_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3115] = ['', ['QCD_Pt_600_800_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3116] = ['', ['QCD_Pt_80_120_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]


workflows[3117] = ['', ['Higgs200ChargedTaus_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3118] = ['', ['JpsiMM_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3119] = ['', ['TTbar_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3120] = ['', ['WE_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3121] = ['', ['ZEE_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3122] = ['', ['ZTT_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3123] = ['', ['H130GGgluonfusion_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3124] = ['', ['PhotonJets_Pt_10_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3125] = ['', ['QQH1352T_Tauola_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]

workflows[3126] = ['', ['MinBias_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3127] = ['', ['WM_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3128] = ['', ['ZMM_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]

workflows[3129] = ['', ['ADDMonoJet_d3MD3_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3130] = ['', ['ZpMM_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3131] = ['', ['WpM_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]


