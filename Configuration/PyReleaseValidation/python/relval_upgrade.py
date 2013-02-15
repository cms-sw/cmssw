
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = {}

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used

#ExtendedPhase1
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
workflows[3110] = ['', ['TTbarLepton_UPGPhase1_8','DIGIUP','RECOUP','HARVESTUP']]
workflows[3111] = ['', ['Wjet_Pt_80_120_UPGPhase1_8','DIGIUP','RECOUP','HARVESTUP']]
workflows[3112] = ['', ['Wjet_Pt_3000_3500_UPGPhase1_8','DIGIUP','RECOUP','HARVESTUP']]
workflows[3113] = ['', ['LM1_sfts_UPGPhase1_8','DIGIUP','RECOUP','HARVESTUP']]

workflows[3114] = ['', ['QCD_Pt_3000_3500_UPGPhase1_8','DIGIUP','RECOUP','HARVESTUP']]
workflows[3115] = ['', ['QCD_Pt_600_800_UPGPhase1_8','DIGIUP','RECOUP','HARVESTUP']]
workflows[3116] = ['', ['QCD_Pt_80_120_UPGPhase1_8','DIGIUP','RECOUP','HARVESTUP']]

workflows[3117] = ['', ['Higgs200ChargedTaus_UPGPhase1_8','DIGIUP','RECOUP','HARVESTUP']]
workflows[3118] = ['', ['JpsiMM_UPGPhase1_8','DIGIUP','RECOUP','HARVESTUP']]
workflows[3119] = ['', ['TTbar_UPGPhase1_8','DIGIUP','RECOUP','HARVESTUP']]
workflows[3120] = ['', ['WE_UPGPhase1_8','DIGIUP','RECOUP','HARVESTUP']]
workflows[3121] = ['', ['ZEE_UPGPhase1_8','DIGIUP','RECOUP','HARVESTUP']]
workflows[3122] = ['', ['ZTT_UPGPhase1_8','DIGIUP','RECOUP','HARVESTUP']]
workflows[3123] = ['', ['H130GGgluonfusion_UPGPhase1_8','DIGIUP','RECOUP','HARVESTUP']]
workflows[3124] = ['', ['PhotonJets_Pt_10_UPGPhase1_8','DIGIUP','RECOUP','HARVESTUP']]
workflows[3125] = ['', ['QQH1352T_Tauola_UPGPhase1_8','DIGIUP','RECOUP','HARVESTUP']]

workflows[3126] = ['', ['MinBias_TuneZ2star_UPGPhase1_8','DIGIUP','RECOUP','HARVESTUP']]
workflows[3127] = ['', ['WM_UPGPhase1_8','DIGIUP','RECOUP','HARVESTUP']]
workflows[3128] = ['', ['ZMM_UPGPhase1_8','DIGIUP','RECOUP','HARVESTUP']]

workflows[3129] = ['', ['ADDMonoJet_d3MD3_UPGPhase1_8','DIGIUP','RECOUP','HARVESTUP']]
workflows[3130] = ['', ['ZpMM_UPGPhase1_8','DIGIUP','RECOUP','HARVESTUP']]
workflows[3131] = ['', ['WpM_UPGPhase1_8','DIGIUP','RECOUP','HARVESTUP']]


#std wf @14TeV
#workflows[3132] = ['', ['TTbarLepton_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']] #no gen fragment
workflows[3133] = ['', ['Wjet_Pt_80_120_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3134] = ['', ['Wjet_Pt_3000_3500_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3135] = ['', ['LM1_sfts_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]


workflows[3136] = ['', ['QCD_Pt_3000_3500_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
#workflows[3137] = ['', ['QCD_Pt_600_800_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']] #no gen fragment
workflows[3138] = ['', ['QCD_Pt_80_120_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]


workflows[3139] = ['', ['Higgs200ChargedTaus_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3140] = ['', ['JpsiMM_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3141] = ['', ['TTbar_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3142] = ['', ['WE_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3143] = ['', ['ZEE_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3144] = ['', ['ZTT_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3145] = ['', ['H130GGgluonfusion_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3146] = ['', ['PhotonJets_Pt_10_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3147] = ['', ['QQH1352T_Tauola_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]

workflows[3148] = ['', ['MinBias_TuneZ2star_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3149] = ['', ['WM_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]
workflows[3150] = ['', ['ZMM_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]

#workflows[3151] = ['', ['ADDMonoJet_d3MD3_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]#no gen fragment
#workflows[3152] = ['', ['ZpMM_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]#no gen fragment
#workflows[3153] = ['', ['WpM_UPGPhase1_14','DIGIUP','RECOUP','HARVESTUP']]#no gen fragment


#2015
#part gun
workflows[3200] = ['', ['FourMuPt1_200_UPG2015','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3201] = ['', ['SingleElectronPt10_UPG2015','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3202] = ['', ['SingleElectronPt1000_UPG2015','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3203] = ['', ['SingleElectronPt35_UPG2015','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3204] = ['', ['SingleGammaPt10_UPG2015','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3205] = ['', ['SingleGammaPt35_UPG2015','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3206] = ['', ['SingleMuPt1_UPG2015','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3207] = ['', ['SingleMuPt10_UPG2015','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3208] = ['', ['SingleMuPt100_UPG2015','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3209] = ['', ['SingleMuPt1000_UPG2015','DIGIUP15','RECOUP15','HARVESTUP15']]

#std wf @8TeV
workflows[3210] = ['', ['TTbarLepton_UPG2015_8','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3211] = ['', ['Wjet_Pt_80_120_UPG2015_8','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3212] = ['', ['Wjet_Pt_3000_3500_UPG2015_8','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3213] = ['', ['LM1_sfts_UPG2015_8','DIGIUP15','RECOUP15','HARVESTUP15']]

workflows[3214] = ['', ['QCD_Pt_3000_3500_UPG2015_8','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3215] = ['', ['QCD_Pt_600_800_UPG2015_8','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3216] = ['', ['QCD_Pt_80_120_UPG2015_8','DIGIUP15','RECOUP15','HARVESTUP15']]

workflows[3217] = ['', ['Higgs200ChargedTaus_UPG2015_8','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3218] = ['', ['JpsiMM_UPG2015_8','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3219] = ['', ['TTbar_UPG2015_8','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3220] = ['', ['WE_UPG2015_8','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3221] = ['', ['ZEE_UPG2015_8','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3222] = ['', ['ZTT_UPG2015_8','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3223] = ['', ['H130GGgluonfusion_UPG2015_8','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3224] = ['', ['PhotonJets_Pt_10_UPG2015_8','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3225] = ['', ['QQH1352T_Tauola_UPG2015_8','DIGIUP15','RECOUP15','HARVESTUP15']]

workflows[3226] = ['', ['MinBias_TuneZ2star_UPG2015_8','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3227] = ['', ['WM_UPG2015_8','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3228] = ['', ['ZMM_UPG2015_8','DIGIUP15','RECOUP15','HARVESTUP15']]

workflows[3229] = ['', ['ADDMonoJet_d3MD3_UPG2015_8','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3230] = ['', ['ZpMM_UPG2015_8','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3231] = ['', ['WpM_UPG2015_8','DIGIUP15','RECOUP15','HARVESTUP15']]


#std wf @14TeV
#workflows[3232] = ['', ['TTbarLepton_UPG2015_14','DIGIUP15','RECOUP15','HARVESTUP15']] #no gen fragment
workflows[3233] = ['', ['Wjet_Pt_80_120_UPG2015_14','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3234] = ['', ['Wjet_Pt_3000_3500_UPG2015_14','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3235] = ['', ['LM1_sfts_UPG2015_14','DIGIUP15','RECOUP15','HARVESTUP15']]


workflows[3236] = ['', ['QCD_Pt_3000_3500_UPG2015_14','DIGIUP15','RECOUP15','HARVESTUP15']]
#workflows[3237] = ['', ['QCD_Pt_600_800_UPG2015_14','DIGIUP15','RECOUP15','HARVESTUP15']] #no gen fragment
workflows[3238] = ['', ['QCD_Pt_80_120_UPG2015_14','DIGIUP15','RECOUP15','HARVESTUP15']]


workflows[3239] = ['', ['Higgs200ChargedTaus_UPG2015_14','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3240] = ['', ['JpsiMM_UPG2015_14','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3241] = ['', ['TTbar_UPG2015_14','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3242] = ['', ['WE_UPG2015_14','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3243] = ['', ['ZEE_UPG2015_14','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3244] = ['', ['ZTT_UPG2015_14','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3245] = ['', ['H130GGgluonfusion_UPG2015_14','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3246] = ['', ['PhotonJets_Pt_10_UPG2015_14','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3247] = ['', ['QQH1352T_Tauola_UPG2015_14','DIGIUP15','RECOUP15','HARVESTUP15']]

workflows[3248] = ['', ['MinBias_TuneZ2star_UPG2015_14','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3249] = ['', ['WM_UPG2015_14','DIGIUP15','RECOUP15','HARVESTUP15']]
workflows[3250] = ['', ['ZMM_UPG2015_14','DIGIUP15','RECOUP15','HARVESTUP15']]

#workflows[3251] = ['', ['ADDMonoJet_d3MD3_UPG2015_14','DIGIUP15','RECOUP15','HARVESTUP15']]#no gen fragment
#workflows[3252] = ['', ['ZpMM_UPG2015_14','DIGIUP15','RECOUP15','HARVESTUP15']]#no gen fragment
#workflows[3253] = ['', ['WpM_UPG2015_14','DIGIUP15','RECOUP15','HARVESTUP15']]#no gen fragment


#2017
#part gun
workflows[3300] = ['', ['FourMuPt1_200_UPG2017','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3301] = ['', ['SingleElectronPt10_UPG2017','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3302] = ['', ['SingleElectronPt1000_UPG2017','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3303] = ['', ['SingleElectronPt35_UPG2017','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3304] = ['', ['SingleGammaPt10_UPG2017','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3305] = ['', ['SingleGammaPt35_UPG2017','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3306] = ['', ['SingleMuPt1_UPG2017','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3307] = ['', ['SingleMuPt10_UPG2017','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3308] = ['', ['SingleMuPt100_UPG2017','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3309] = ['', ['SingleMuPt1000_UPG2017','DIGIUP17','RECOUP17','HARVESTUP17']]

#std wf @8TeV
workflows[3310] = ['', ['TTbarLepton_UPG2017_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3311] = ['', ['Wjet_Pt_80_120_UPG2017_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3312] = ['', ['Wjet_Pt_3000_3500_UPG2017_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3313] = ['', ['LM1_sfts_UPG2017_8','DIGIUP17','RECOUP17','HARVESTUP17']]

workflows[3314] = ['', ['QCD_Pt_3000_3500_UPG2017_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3315] = ['', ['QCD_Pt_600_800_UPG2017_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3316] = ['', ['QCD_Pt_80_120_UPG2017_8','DIGIUP17','RECOUP17','HARVESTUP17']]

workflows[3317] = ['', ['Higgs200ChargedTaus_UPG2017_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3318] = ['', ['JpsiMM_UPG2017_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3319] = ['', ['TTbar_UPG2017_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3320] = ['', ['WE_UPG2017_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3321] = ['', ['ZEE_UPG2017_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3322] = ['', ['ZTT_UPG2017_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3323] = ['', ['H130GGgluonfusion_UPG2017_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3324] = ['', ['PhotonJets_Pt_10_UPG2017_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3325] = ['', ['QQH1352T_Tauola_UPG2017_8','DIGIUP17','RECOUP17','HARVESTUP17']]

workflows[3326] = ['', ['MinBias_TuneZ2star_UPG2017_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3327] = ['', ['WM_UPG2017_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3328] = ['', ['ZMM_UPG2017_8','DIGIUP17','RECOUP17','HARVESTUP17']]

workflows[3329] = ['', ['ADDMonoJet_d3MD3_UPG2017_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3330] = ['', ['ZpMM_UPG2017_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3331] = ['', ['WpM_UPG2017_8','DIGIUP17','RECOUP17','HARVESTUP17']]


#std wf @14TeV
#workflows[3332] = ['', ['TTbarLepton_UPG2017_14','DIGIUP17','RECOUP17','HARVESTUP17']] #no gen fragment
workflows[3333] = ['', ['Wjet_Pt_80_120_UPG2017_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3334] = ['', ['Wjet_Pt_3000_3500_UPG2017_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3335] = ['', ['LM1_sfts_UPG2017_14','DIGIUP17','RECOUP17','HARVESTUP17']]


workflows[3336] = ['', ['QCD_Pt_3000_3500_UPG2017_14','DIGIUP17','RECOUP17','HARVESTUP17']]
#workflows[3337] = ['', ['QCD_Pt_600_800_UPG2017_14','DIGIUP17','RECOUP17','HARVESTUP17']] #no gen fragment
workflows[3338] = ['', ['QCD_Pt_80_120_UPG2017_14','DIGIUP17','RECOUP17','HARVESTUP17']]


workflows[3339] = ['', ['Higgs200ChargedTaus_UPG2017_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3340] = ['', ['JpsiMM_UPG2017_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3341] = ['', ['TTbar_UPG2017_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3342] = ['', ['WE_UPG2017_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3343] = ['', ['ZEE_UPG2017_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3344] = ['', ['ZTT_UPG2017_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3345] = ['', ['H130GGgluonfusion_UPG2017_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3346] = ['', ['PhotonJets_Pt_10_UPG2017_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3347] = ['', ['QQH1352T_Tauola_UPG2017_14','DIGIUP17','RECOUP17','HARVESTUP17']]

workflows[3348] = ['', ['MinBias_TuneZ2star_UPG2017_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3349] = ['', ['WM_UPG2017_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3350] = ['', ['ZMM_UPG2017_14','DIGIUP17','RECOUP17','HARVESTUP17']]

#workflows[3351] = ['', ['ADDMonoJet_d3MD3_UPG2017_14','DIGIUP17','RECOUP17','HARVESTUP17']]#no gen fragment
#workflows[3352] = ['', ['ZpMM_UPG2017_14','DIGIUP17','RECOUP17','HARVESTUP17']]#no gen fragment
#workflows[3353] = ['', ['WpM_UPG2017_14','DIGIUP17','RECOUP17','HARVESTUP17']]#no gen fragment


