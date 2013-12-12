
# import the definition of the steps and input files:
from  Configuration.PyReleaseValidation.relval_steps import *

# here only define the workflows as a combination of the steps defined above:
workflows = Matrix()

# each workflow defines a name and a list of steps to be done. 
# if no explicit name/label given for the workflow (first arg),
# the name of step1 will be used



#2017
#part gun
workflows[3300.0]  = [ 'FourMuPt1_200' ,  ['FourMuPt1_200_UPG2017EcalFine','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3301.0]  = [ 'SingleElectronPt10' ,  ['SingleElectronPt10_UPG2017EcalFine','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3302.0]  = [ 'SingleElectronPt1000' ,  ['SingleElectronPt1000_UPG2017EcalFine','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3303.0]  = [ 'SingleElectronPt35' ,  ['SingleElectronPt35_UPG2017EcalFine','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3304.0]  = [ 'SingleGammaPt10' ,  ['SingleGammaPt10_UPG2017EcalFine','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3305.0]  = [ 'SingleGammaPt35' ,  ['SingleGammaPt35_UPG2017EcalFine','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3306.0]  = [ 'SingleMuPt1' ,  ['SingleMuPt1_UPG2017EcalFine','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3307.0]  = [ 'SingleMuPt10' ,  ['SingleMuPt10_UPG2017EcalFine','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3308.0]  = [ 'SingleMuPt100' ,  ['SingleMuPt100_UPG2017EcalFine','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3309.0]  = [ 'SingleMuPt1000' ,  ['SingleMuPt1000_UPG2017EcalFine','DIGIUP17','RECOUP17','HARVESTUP17']]

#std wf @8TeV
workflows[3310.0]  = [ 'TTbarLepton_8TeV' ,  ['TTbarLepton_UPG2017EcalFine_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3311.0]  = [ 'Wjet_Pt_80_120_8TeV' ,  ['Wjet_Pt_80_120_UPG2017EcalFine_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3312.0]  = [ 'Wjet_Pt_3000_3500_8TeV' ,  ['Wjet_Pt_3000_3500_UPG2017EcalFine_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3313.0]  = [ 'LM1_sfts_8TeV' ,  ['LM1_sfts_UPG2017EcalFine_8','DIGIUP17','RECOUP17','HARVESTUP17']]

workflows[3314.0]  = [ 'QCD_Pt_3000_3500_8TeV' ,  ['QCD_Pt_3000_3500_UPG2017EcalFine_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3315.0]  = [ 'QCD_Pt_600_800_8TeV' ,  ['QCD_Pt_600_800_UPG2017EcalFine_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3316.0]  = [ 'QCD_Pt_80_120_8TeV' ,  ['QCD_Pt_80_120_UPG2017EcalFine_8','DIGIUP17','RECOUP17','HARVESTUP17']]

workflows[3317.0]  = [ 'Higgs200ChargedTaus_8TeV' ,  ['Higgs200ChargedTaus_UPG2017EcalFine_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3318.0]  = [ 'JpsiMM_8TeV' ,  ['JpsiMM_UPG2017EcalFine_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3319.0]  = [ 'TTbar_8TeV' ,  ['TTbar_UPG2017EcalFine_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3320.0]  = [ 'WE_8TeV' ,  ['WE_UPG2017EcalFine_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3321.0]  = [ 'ZEE_8TeV' ,  ['ZEE_UPG2017EcalFine_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3322.0]  = [ 'ZTT_8TeV' ,  ['ZTT_UPG2017EcalFine_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3323.0]  = [ 'H130GGgluonfusion_8TeV' ,  ['H130GGgluonfusion_UPG2017EcalFine_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3324.0]  = [ 'PhotonJets_Pt_10_8TeV' ,  ['PhotonJets_Pt_10_UPG2017EcalFine_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3325.0]  = [ 'QQH1352T_Tauola_8TeV' ,  ['QQH1352T_Tauola_UPG2017EcalFine_8','DIGIUP17','RECOUP17','HARVESTUP17']]

workflows[3326.0]  = [ 'MinBias_TuneZ2star_8TeV' ,  ['MinBias_TuneZ2star_UPG2017EcalFine_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3327.0]  = [ 'WM_8TeV' ,  ['WM_UPG2017EcalFine_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3328.0]  = [ 'ZMM_8TeV' ,  ['ZMM_UPG2017EcalFine_8','DIGIUP17','RECOUP17','HARVESTUP17']]

workflows[3329.0]  = [ 'ADDMonoJet_d3MD3_8TeV' ,  ['ADDMonoJet_d3MD3_UPG2017EcalFine_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3330.0]  = [ 'ZpMM_8TeV' ,  ['ZpMM_UPG2017EcalFine_8','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3331.0]  = [ 'WpM_8TeV' ,  ['WpM_UPG2017EcalFine_8','DIGIUP17','RECOUP17','HARVESTUP17']]



#std wf @14TeV
workflows[3332.0]  = [ 'TTbarLepton_14TeV' ,  ['TTbarLepton_UPG2017EcalFine_14','DIGIUP17','RECOUP17','HARVESTUP17']] #no gen fragment
workflows[3333.0]  = [ 'Wjet_Pt_80_120_14TeV' ,  ['Wjet_Pt_80_120_UPG2017EcalFine_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3334.0]  = [ 'Wjet_Pt_3000_3500_14TeV' ,  ['Wjet_Pt_3000_3500_UPG2017EcalFine_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3335.0]  = [ 'LM1_sfts_14TeV' ,  ['LM1_sfts_UPG2017EcalFine_14','DIGIUP17','RECOUP17','HARVESTUP17']]

workflows[3336.0]  = [ 'QCD_Pt_3000_3500_14TeV' ,  ['QCD_Pt_3000_3500_UPG2017EcalFine_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3337.0]  = [ 'QCD_Pt_600_800_14TeV' ,  ['QCD_Pt_600_800_UPG2017EcalFine_14','DIGIUP17','RECOUP17','HARVESTUP17']] #no gen fragment
workflows[3338.0]  = [ 'QCD_Pt_80_120_14TeV' ,  ['QCD_Pt_80_120_UPG2017EcalFine_14','DIGIUP17','RECOUP17','HARVESTUP17']]

workflows[3339.0]  = [ 'Higgs200ChargedTaus_14TeV' ,  ['Higgs200ChargedTaus_UPG2017EcalFine_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3340.0]  = [ 'JpsiMM_14TeV' ,  ['JpsiMM_UPG2017EcalFine_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3341.0]  = [ 'TTbar_14TeV' ,  ['TTbar_UPG2017EcalFine_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3342.0]  = [ 'WE_14TeV' ,  ['WE_UPG2017EcalFine_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3343.0]  = [ 'ZEE_14TeV' ,  ['ZEE_UPG2017EcalFine_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3344.0]  = [ 'ZTT_14TeV' ,  ['ZTT_UPG2017EcalFine_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3345.0]  = [ 'H130GGgluonfusion_14TeV' ,  ['H130GGgluonfusion_UPG2017EcalFine_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3346.0]  = [ 'PhotonJets_Pt_10_14TeV' ,  ['PhotonJets_Pt_10_UPG2017EcalFine_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3347.0]  = [ 'QQH1352T_Tauola_14TeV' ,  ['QQH1352T_Tauola_UPG2017EcalFine_14','DIGIUP17','RECOUP17','HARVESTUP17']]

workflows[3348.0]  = [ 'MinBias_TuneZ2star_14TeV' ,  ['MinBias_TuneZ2star_UPG2017EcalFine_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3349.0]  = [ 'WM_14TeV' ,  ['WM_UPG2017EcalFine_14','DIGIUP17','RECOUP17','HARVESTUP17']]
workflows[3350.0]  = [ 'ZMM_14TeV' ,  ['ZMM_UPG2017EcalFine_14','DIGIUP17','RECOUP17','HARVESTUP17']]

workflows[3351.0]  = [ 'ADDMonoJet_d3MD3_14TeV' ,  ['ADDMonoJet_d3MD3_UPG2017EcalFine_14','DIGIUP17','RECOUP17','HARVESTUP17']]#no gen fragment
workflows[3352.0]  = [ 'ZpMM_14TeV' ,  ['ZpMM_UPG2017EcalFine_14','DIGIUP17','RECOUP17','HARVESTUP17']]#no gen fragment
workflows[3353.0]  = [ 'WpM_14TeV' ,  ['WpM_UPG2017EcalFine_14','DIGIUP17','RECOUP17','HARVESTUP17']]#no gen fragment


#2017 PU
#part gun
workflows[3300.1]  = [ 'FourMuPt1_200' ,  ['FourMuPt1_200_UPG2017EcalFine','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3301.1]  = [ 'SingleElectronPt10' ,  ['SingleElectronPt10_UPG2017EcalFine','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3302.1]  = [ 'SingleElectronPt1000' ,  ['SingleElectronPt1000_UPG2017EcalFine','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3303.1]  = [ 'SingleElectronPt35' ,  ['SingleElectronPt35_UPG2017EcalFine','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3304.1]  = [ 'SingleGammaPt10' ,  ['SingleGammaPt10_UPG2017EcalFine','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3305.1]  = [ 'SingleGammaPt35' ,  ['SingleGammaPt35_UPG2017EcalFine','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3306.1]  = [ 'SingleMuPt1' ,  ['SingleMuPt1_UPG2017EcalFine','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3307.1]  = [ 'SingleMuPt10' ,  ['SingleMuPt10_UPG2017EcalFine','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3308.1]  = [ 'SingleMuPt100' ,  ['SingleMuPt100_UPG2017EcalFine','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3309.1]  = [ 'SingleMuPt1000' ,  ['SingleMuPt1000_UPG2017EcalFine','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]

#std wf @8TeV
workflows[3310.1]  = [ 'TTbarLepton_8TeV' ,  ['TTbarLepton_UPG2017EcalFine_8','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3311.1]  = [ 'Wjet_Pt_80_120_8TeV' ,  ['Wjet_Pt_80_120_UPG2017EcalFine_8','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3312.1]  = [ 'Wjet_Pt_3000_3500_8TeV' ,  ['Wjet_Pt_3000_3500_UPG2017EcalFine_8','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3313.1]  = [ 'LM1_sfts_8TeV' ,  ['LM1_sfts_UPG2017EcalFine_8','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]

workflows[3314.1]  = [ 'QCD_Pt_3000_3500_8TeV' ,  ['QCD_Pt_3000_3500_UPG2017EcalFine_8','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3315.1]  = [ 'QCD_Pt_600_800_8TeV' ,  ['QCD_Pt_600_800_UPG2017EcalFine_8','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3316.1]  = [ 'QCD_Pt_80_120_8TeV' ,  ['QCD_Pt_80_120_UPG2017EcalFine_8','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]

workflows[3317.1]  = [ 'Higgs200ChargedTaus_8TeV' ,  ['Higgs200ChargedTaus_UPG2017EcalFine_8','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3318.1]  = [ 'JpsiMM_8TeV' ,  ['JpsiMM_UPG2017EcalFine_8','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3319.1]  = [ 'TTbar_8TeV' ,  ['TTbar_UPG2017EcalFine_8','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3320.1]  = [ 'WE_8TeV' ,  ['WE_UPG2017EcalFine_8','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3321.1]  = [ 'ZEE_8TeV' ,  ['ZEE_UPG2017EcalFine_8','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3322.1]  = [ 'ZTT_8TeV' ,  ['ZTT_UPG2017EcalFine_8','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3323.1]  = [ 'H130GGgluonfusion_8TeV' ,  ['H130GGgluonfusion_UPG2017EcalFine_8','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3324.1]  = [ 'PhotonJets_Pt_10_8TeV' ,  ['PhotonJets_Pt_10_UPG2017EcalFine_8','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3325.1]  = [ 'QQH1352T_Tauola_8TeV' ,  ['QQH1352T_Tauola_UPG2017EcalFine_8','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]

workflows[3326.1]  = [ 'MinBias_TuneZ2star_8TeV' ,  ['MinBias_TuneZ2star_UPG2017EcalFine_8','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3327.1]  = [ 'WM_8TeV' ,  ['WM_UPG2017EcalFine_8','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3328.1]  = [ 'ZMM_8TeV' ,  ['ZMM_UPG2017EcalFine_8','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]

workflows[3329.1]  = [ 'ADDMonoJet_d3MD3_8TeV' ,  ['ADDMonoJet_d3MD3_UPG2017EcalFine_8','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3330.1]  = [ 'ZpMM_8TeV' ,  ['ZpMM_UPG2017EcalFine_8','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3331.1]  = [ 'WpM_8TeV' ,  ['WpM_UPG2017EcalFine_8','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]



#std wf @14TeV
workflows[3332.1]  = [ 'TTbarLepton_14TeV' ,  ['TTbarLepton_UPG2017EcalFine_14','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']] #no gen fragment
workflows[3333.1]  = [ 'Wjet_Pt_80_120_14TeV' ,  ['Wjet_Pt_80_120_UPG2017EcalFine_14','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3334.1]  = [ 'Wjet_Pt_3000_3500_14TeV' ,  ['Wjet_Pt_3000_3500_UPG2017EcalFine_14','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3335.1]  = [ 'LM1_sfts_14TeV' ,  ['LM1_sfts_UPG2017EcalFine_14','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]

workflows[3336.1]  = [ 'QCD_Pt_3000_3500_14TeV' ,  ['QCD_Pt_3000_3500_UPG2017EcalFine_14','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3337.1]  = [ 'QCD_Pt_600_800_14TeV' ,  ['QCD_Pt_600_800_UPG2017EcalFine_14','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']] #no gen fragment
workflows[3338.1]  = [ 'QCD_Pt_80_120_14TeV' ,  ['QCD_Pt_80_120_UPG2017EcalFine_14','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]

workflows[3339.1]  = [ 'Higgs200ChargedTaus_14TeV' ,  ['Higgs200ChargedTaus_UPG2017EcalFine_14','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3340.1]  = [ 'JpsiMM_14TeV' ,  ['JpsiMM_UPG2017EcalFine_14','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3341.1]  = [ 'TTbar_14TeV' ,  ['TTbar_UPG2017EcalFine_14','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3342.1]  = [ 'WE_14TeV' ,  ['WE_UPG2017EcalFine_14','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3343.1]  = [ 'ZEE_14TeV' ,  ['ZEE_UPG2017EcalFine_14','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3344.1]  = [ 'ZTT_14TeV' ,  ['ZTT_UPG2017EcalFine_14','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3345.1]  = [ 'H130GGgluonfusion_14TeV' ,  ['H130GGgluonfusion_UPG2017EcalFine_14','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3346.1]  = [ 'PhotonJets_Pt_10_14TeV' ,  ['PhotonJets_Pt_10_UPG2017EcalFine_14','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3347.1]  = [ 'QQH1352T_Tauola_14TeV' ,  ['QQH1352T_Tauola_UPG2017EcalFine_14','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]

workflows[3348.1]  = [ 'MinBias_TuneZ2star_14TeV' ,  ['MinBias_TuneZ2star_UPG2017EcalFine_14','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3349.1]  = [ 'WM_14TeV' ,  ['WM_UPG2017EcalFine_14','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]
workflows[3350.1]  = [ 'ZMM_14TeV' ,  ['ZMM_UPG2017EcalFine_14','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]

workflows[3351.1]  = [ 'ADDMonoJet_d3MD3_14TeV' ,  ['ADDMonoJet_d3MD3_UPG2017EcalFine_14','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]#no gen fragment
workflows[3352.1]  = [ 'ZpMM_14TeV' ,  ['ZpMM_UPG2017EcalFine_14','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]#no gen fragment
workflows[3353.1]  = [ 'WpM_14TeV' ,  ['WpM_UPG2017EcalFine_14','DIGIPUUP17ECALFINE','RECOPUUP17ECALFINE']]#no gen fragment



