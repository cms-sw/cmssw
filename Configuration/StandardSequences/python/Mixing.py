Mixing = {}


def addMixingScenario(label,dict):
    global Mixing
    if label in Mixing:
        print 'duplicated definition of',label
    else:
        #try:
        #    m=__import__(dict['file'])
        #except:
        #    raise Exception('no file'+dict['file']+'to be loaded')
        Mixing[label]=dict

##full sim section
addMixingScenario("156BxLumiPileUp",{'file': 'SimGeneral.MixingModule.StageA156Bx_cfi'})
addMixingScenario("E10TeV_FIX_1_BX432",{'file': 'SimGeneral.MixingModule.mix_E10TeV_FIX_1_BX432_cfi'})
addMixingScenario("E10TeV_FIX_2_BX432",{'file': 'SimGeneral.MixingModule.mix_E10TeV_FIX_1_BX432_cfi', 'N': 2})
addMixingScenario("E10TeV_FIX_3_BX432",{'file': 'SimGeneral.MixingModule.mix_E10TeV_FIX_1_BX432_cfi', 'N': 3})
addMixingScenario("E10TeV_FIX_5_BX432",{'file': 'SimGeneral.MixingModule.mix_E10TeV_FIX_1_BX432_cfi', 'N': 5})
addMixingScenario("E10TeV_L13E31_BX432",{'file': 'SimGeneral.MixingModule.mix_E10TeV_L13E31_BX432_cfi'})
addMixingScenario("E10TeV_L21E31_BX432",{'file': 'SimGeneral.MixingModule.mix_E10TeV_L21E31_BX432_cfi'})
addMixingScenario("E14TeV_L10E33_BX2808",{'file': 'SimGeneral.MixingModule.mix_E14TeV_L10E33_BX2808_cfi'})
addMixingScenario("E14TeV_L28E32_BX2808",{'file': 'SimGeneral.MixingModule.mix_E14TeV_L28E32_BX2808_cfi'})
addMixingScenario("E7TeV_AVE_01_BX2808",{'file': 'SimGeneral.MixingModule.mix_E7TeV_AVE_2_BX2808_cfi', 'N': 0.1})
addMixingScenario("E7TeV_AVE_02_BX2808",{'file': 'SimGeneral.MixingModule.mix_E7TeV_AVE_2_BX2808_cfi', 'N': 0.2})
addMixingScenario("E7TeV_AVE_05_BX2808",{'file': 'SimGeneral.MixingModule.mix_E7TeV_AVE_2_BX2808_cfi', 'N': 0.5})
addMixingScenario("E7TeV_AVE_10_BX2808",{'file': 'SimGeneral.MixingModule.mix_E7TeV_AVE_2_BX2808_cfi', 'N': 10})
addMixingScenario("E7TeV_AVE_1_BX156",{'file': 'SimGeneral.MixingModule.mix_E7TeV_AVE_1_BX156_cfi', 'N': 1})
addMixingScenario("E7TeV_AVE_1_BX2808",{'file': 'SimGeneral.MixingModule.mix_E7TeV_AVE_2_BX2808_cfi', 'N': 1})
addMixingScenario("E7TeV_AVE_20_BX2808",{'file': 'SimGeneral.MixingModule.mix_E7TeV_AVE_2_BX2808_cfi', 'N': 20})
addMixingScenario("E7TeV_AVE_2_8_BX_50ns",{'file': 'SimGeneral.MixingModule.mix_E7TeV_AVE_2_8_BX_50ns_cfi'})
addMixingScenario("E7TeV_AVE_2_8_BXgt50ns_intime_only",{'file': 'SimGeneral.MixingModule.mix_E7TeV_AVE_2_8_BXgt50ns_intime_only_cfi'})
addMixingScenario("E7TeV_AVE_2_BX156",{'file': 'SimGeneral.MixingModule.mix_E7TeV_AVE_1_BX156_cfi','N': 2})
addMixingScenario("E7TeV_AVE_2_BX2808",{'file': 'SimGeneral.MixingModule.mix_E7TeV_AVE_2_BX2808_cfi'})
addMixingScenario("E7TeV_AVE_3_BX156",{'file': 'SimGeneral.MixingModule.mix_E7TeV_AVE_1_BX156_cfi','N': 3})
addMixingScenario("E7TeV_AVE_50_BX2808",{'file': 'SimGeneral.MixingModule.mix_E7TeV_AVE_2_BX2808_cfi','N': 50})
addMixingScenario("E7TeV_AVE_5_BX156",{'file': 'SimGeneral.MixingModule.mix_E7TeV_AVE_1_BX156_cfi', 'N': 5})
addMixingScenario("E7TeV_AVE_5_BX2808",{'file': 'SimGeneral.MixingModule.mix_E7TeV_AVE_2_BX2808_cfi', 'N': 5})
addMixingScenario("E7TeV_FIX_1_BX156",{'file': 'SimGeneral.MixingModule.mix_E7TeV_FIX_1_BX156_cfi'})
addMixingScenario("E7TeV_FIX_2_BX156",{'file': 'SimGeneral.MixingModule.mix_E7TeV_FIX_1_BX156_cfi', 'N': 2})
addMixingScenario("E7TeV_FIX_3_BX156",{'file': 'SimGeneral.MixingModule.mix_E7TeV_FIX_1_BX156_cfi', 'N': 3})
addMixingScenario("E7TeV_FIX_5_BX156",{'file': 'SimGeneral.MixingModule.mix_E7TeV_FIX_1_BX156_cfi', 'N': 5})
addMixingScenario("E7TeV_L34E30_BX156",{'file': 'SimGeneral.MixingModule.mix_E7TeV_L34E30_BX156_cfi'})
addMixingScenario("E7TeV_L69E30_BX156",{'file': 'SimGeneral.MixingModule.mix_E7TeV_L69E30_BX156_cfi'})
addMixingScenario("E8TeV_AVE_4_BX_50ns",{'file':'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':50,'B': (-3,2),'N': 4})
addMixingScenario("E8TeV_AVE_10_BX_50ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':50, 'B': (-3,2), 'N': 10})
addMixingScenario("E8TeV_AVE_10_BX_25ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':25, 'B': (-3,2), 'N': 10})
addMixingScenario("E8TeV_AVE_16_BX_25ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi'})
addMixingScenario("E8TeV_AVE_16_BX_50ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX': 50, 'B': (-3,2)})
addMixingScenario("E8TeV_AVE_10_BX_50ns_300ns_spread",{'file':'SimGeneral.MixingModule.mix_E8TeV_AVE_10_BX_50ns_300ns_spread_cfi'})
addMixingScenario("E8TeV_AVE_10_BX_25ns_300ns_spread",{'file':'SimGeneral.MixingModule.mix_E8TeV_AVE_10_BX_25ns_300ns_spread_cfi'})
addMixingScenario("HiMix",{'file': 'SimGeneral.MixingModule.HiEventMixing_cff'})
addMixingScenario("HighLumiPileUp",{'file': 'SimGeneral.MixingModule.mixHighLumPU_cfi'})
addMixingScenario("InitialLumiPileUp",{'file': 'SimGeneral.MixingModule.mixInitialLumPU_cfi'})
addMixingScenario("LowLumiPileUp",{'file': 'SimGeneral.MixingModule.mixLowLumPU_cfi'})
addMixingScenario("LowLumiPileUp4Sources",{'file': 'SimGeneral.MixingModule.mixLowLumPU_4sources_cfi'})
addMixingScenario("LowLumiPileUp4Sources_ProdStep1",{'file': 'SimGeneral.MixingModule.mixLowLumPU_4sources_mixProdStep1_cfi'})
addMixingScenario("LowLumiPileUp_ProdStep1",{'file': 'SimGeneral.MixingModule.mixLowLumPU_mixProdStep1_cfi'})
addMixingScenario("NoPileUp",{'file': 'SimGeneral.MixingModule.mixNoPU_cfi'})
addMixingScenario("E7TeV_ProbDist_2010Data_BX156",{'file': 'SimGeneral.MixingModule.mix_E7TeV_ProbDist_2010Data_BX156_cfi'})
addMixingScenario("E8TeV_ProbDist_2011EarlyData_50ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_ProbDist_2011EarlyData_50ns_cfi'})
addMixingScenario("E8TeV_FlatDist_2011EarlyData_50ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_FlatDist_2011EarlyData_50ns_cfi'})
addMixingScenario("E7TeV_FlatDist10_2011EarlyData_50ns",{'file': 'SimGeneral.MixingModule.mix_E7TeV_FlatDist10_2011EarlyData_50ns_cfi'})
addMixingScenario("E7TeV_FlatDist10_2011EarlyData_75ns",{'file': 'SimGeneral.MixingModule.mix_E7TeV_FlatDist10_2011EarlyData_75ns_cfi'})
addMixingScenario("E7TeV_FlatDist10_2011EarlyData_inTimeOnly",{'file': 'SimGeneral.MixingModule.mix_E7TeV_FlatDist10_2011EarlyData_inTimeOnly_cfi'})
addMixingScenario("E7TeV_Flat20_AllEarly_75ns",{'file': 'SimGeneral.MixingModule.mix_E7TeV_Flat20_AllEarly_75ns_cfi'})
addMixingScenario("E7TeV_Flat20_AllLate_75ns",{'file': 'SimGeneral.MixingModule.mix_E7TeV_Flat20_AllLate_75ns_cfi'})
addMixingScenario("E7TeV_Flat20_AllEarly_50ns",{'file': 'SimGeneral.MixingModule.mix_E7TeV_Flat20_AllEarly_50ns_cfi'})
addMixingScenario("E7TeV_Flat20_AllLate_50ns",{'file': 'SimGeneral.MixingModule.mix_E7TeV_Flat20_AllLate_50ns_cfi'})
addMixingScenario("E7TeV_FlatDist10_2011EarlyData_50ns_PoissonOOT",{'file': 'SimGeneral.MixingModule.mix_E7TeV_FlatDist10_2011EarlyData_50ns_PoissonOOT'})
addMixingScenario("E7TeV_FlatDist10_2011EarlyData_25ns_PoissonOOT",{'file': 'SimGeneral.MixingModule.mix_E7TeV_FlatDist10_2011EarlyData_25ns_PoissonOOT_cfi'})
addMixingScenario("E7TeV_Ave18p4_50ns", {'file': 'SimGeneral.MixingModule.mix_E7TeV_Ave18p4_50ns_cfi'})
addMixingScenario("E7TeV_Ave23_50ns", {'file': 'SimGeneral.MixingModule.mix_E7TeV_Ave18p4_50ns_cfi', 'N': 23 })
addMixingScenario("E7TeV_Ave32_50ns", {'file': 'SimGeneral.MixingModule.mix_E7TeV_Ave18p4_50ns_cfi', 'N': 32 })
addMixingScenario("E7TeV_Ave25_50ns_PoissonOOTPU",{'file': 'SimGeneral.MixingModule.mix_E7TeV_Ave25_50ns_PoissonOOTPU_cfi'})
addMixingScenario("E7TeV_Ave25_25ns_PoissonOOTPU",{'file': 'SimGeneral.MixingModule.mix_E7TeV_Ave25_25ns_PoissonOOTPU_cfi'})
addMixingScenario("E7TeV_Fall2011ReDigi_prelim_50ns_PoissonOOT",{'file': 'SimGeneral.MixingModule.mix_E7TeV_Fall2011ReDigi_prelim_50ns_PoissonOOT_cfi'})
addMixingScenario("E7TeV_Fall2011ReDigi_50ns_PoissonOOT",{'file': 'SimGeneral.MixingModule.mix_E7TeV_Fall2011ReDigi_50ns_PoissonOOT_cfi'})
addMixingScenario("E7TeV_Fall2011ReDigi_25ns_PoissonOOT",{'file': 'SimGeneral.MixingModule.mix_E7TeV_Fall2011ReDigi_25ns_PoissonOOT_cfi'})
addMixingScenario("E7TeV_Fall2011_Reprocess_50ns_PoissonOOTPU",{'file': 'SimGeneral.MixingModule.mix_E7TeV_Fall2011_Reprocess_50ns_PoissonOOTPU_cfi'})
addMixingScenario("E7TeV_Chamonix2012_50ns_PoissonOOT",{'file': 'SimGeneral.MixingModule.mix_E7TeV_Chamonix2012_50ns_PoissonOOT_cfi'})
addMixingScenario("2012_lumiLevel_15_20_50ns_PoissonOOTPU",{'file': 'SimGeneral.MixingModule.mix_2012_lumiLevel_15_20_50ns_PoissonOOTPU_cfi'})
addMixingScenario("2012_peak11_25ns_PoissonOOTPU",{'file': 'SimGeneral.MixingModule.mix_2012_peak11_25ns_PoissonOOTPU_cfi'})
addMixingScenario("2012_peak26_50ns_PoissonOOTPU",{'file': 'SimGeneral.MixingModule.mix_2012_peak26_50ns_PoissonOOTPU_cfi'})
addMixingScenario("2012_Startup_50ns_PoissonOOTPU",{'file': 'SimGeneral.MixingModule.mix_2012_Startup_50ns_PoissonOOTPU_cfi'})
addMixingScenario("2012_Summer_50ns_PoissonOOTPU",{'file': 'SimGeneral.MixingModule.mix_2012_Summer_50ns_PoissonOOTPU_cfi'})
addMixingScenario("2012A_Profile_PoissonOOTPU",{'file': 'SimGeneral.MixingModule.mix_2012A_Profile_PoissonOOTPU_cfi'})
addMixingScenario("2012B_Profile_PoissonOOTPU",{'file': 'SimGeneral.MixingModule.mix_2012B_Profile_PoissonOOTPU_cfi'})
addMixingScenario("2012C_Profile_PoissonOOTPU",{'file': 'SimGeneral.MixingModule.mix_2012C_Profile_PoissonOOTPU_cfi'})
addMixingScenario("2012D_Profile_PoissonOOTPU",{'file': 'SimGeneral.MixingModule.mix_2012D_Profile_PoissonOOTPU_cfi'})
addMixingScenario("2011_FinalDist_OOTPU",{'file': 'SimGeneral.MixingModule.mix_2011_FinalDist_OOTPU_cfi'})
addMixingScenario("E8TeV_2012_25nsRunning_PoissonOOTPU",{'file': 'SimGeneral.MixingModule.mix_E8TeV_2012_25nsRunning_PoissonOOTPU_cfi'})
addMixingScenario("E8TeV_2012_25nsRunning_TrainBackOOTPU",{'file': 'SimGeneral.MixingModule.mix_E8TeV_2012_25nsRunning_TrainBackOOTPU_cfi'})
addMixingScenario("E8TeV_2012_25nsRunning_TrainFrontOOTPU",{'file': 'SimGeneral.MixingModule.mix_E8TeV_2012_25nsRunning_TrainFrontOOTPU_cfi'})
addMixingScenario("2012_Summer_50ns_PoissonOOTPU_FixedInTime0",{'file': 'SimGeneral.MixingModule.mix_2012_Summer_50ns_PoissonOOTPU_FixedInTime0_cfi'})
addMixingScenario("2012_Summer_50ns_PoissonOOTPU_FixedInTime30",{'file': 'SimGeneral.MixingModule.mix_2012_Summer_50ns_PoissonOOTPU_FixedInTime30_cfi'})
addMixingScenario("ProdStep2",{'file': 'SimGeneral.MixingModule.mixProdStep2_cfi'})
addMixingScenario("fromDB",{'file': 'SimGeneral.MixingModule.mix_fromDB_cfi'})
##fastsim section
addMixingScenario("FS_NoPileUp",{'file': 'FastSimulation.PileUpProducer.PileUpSimulator_NoPileUp_cff'})
addMixingScenario("FS_LowLumiPileUp",{'file': 'FastSimulation.PileUpProducer.PileUpSimulator_LowLumiPileUp_cff'})
addMixingScenario("FS_FlatDist10_2011EarlyData_inTimeOnly",{'file': 'FastSimulation.PileUpProducer.PileUpSimulator_FlatDist10_2011EarlyData_inTimeOnly_cff'})
addMixingScenario("FS_E7TeV_Fall2011_Reprocess_inTimeOnly",{'file': 'FastSimulation.PileUpProducer.PileUpSimulator_E7TeV_Fall2011_Reprocess_inTimeOnly_cff'})
addMixingScenario("FS_E7TeV_ProbDist_2011Data_inTimeOnly",{'file': 'FastSimulation.PileUpProducer.PileUpSimulator_E7TeV_ProbDist_2011Data_inTimeOnly_cff'})
addMixingScenario("FS_2012_Startup_inTimeOnly",{'file': 'FastSimulation.PileUpProducer.PileUpSimulator_2012_Startup_inTimeOnly_cff'})
addMixingScenario("FS_2012_Summer_inTimeOnly",{'file': 'FastSimulation.PileUpProducer.PileUpSimulator_2012_Summer_inTimeOnly_cff'})
addMixingScenario("FS_mix_2012_Startup_inTimeOnly",{'file': 'FastSimulation.PileUpProducer.mix_2012_Startup_inTimeOnly_cff'})
addMixingScenario("FS_mix_2012_Summer_inTimeOnly",{'file': 'FastSimulation.PileUpProducer.mix_2012_Summer_inTimeOnly_cff'})


#scenarios for L1 tdr work
addMixingScenario("AVE_20_BX_50ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':50, 'B': (-12,3), 'N': 20})
addMixingScenario("AVE_20_BX_25ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':25, 'B': (-12,3), 'N': 20})
addMixingScenario("AVE_25_BX_50ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':50, 'B': (-12,3), 'N': 25})
addMixingScenario("AVE_25_BX_25ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':25, 'B': (-12,3), 'N': 25})
addMixingScenario("AVE_35_BX_50ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':50, 'B': (-12,3), 'N': 35})
addMixingScenario("AVE_35_BX_25ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':25, 'B': (-12,3), 'N': 35})
addMixingScenario("AVE_45_BX_25ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':25, 'B': (-12,3), 'N': 45})
addMixingScenario("AVE_50_BX_50ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':50, 'B': (-12,3), 'N': 50})
addMixingScenario("AVE_50_BX_25ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':25, 'B': (-12,3), 'N': 50})
addMixingScenario("AVE_70_BX_50ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':50, 'B': (-12,3), 'N': 70})
addMixingScenario("AVE_70_BX_25ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':25, 'B': (-12,3), 'N': 70})
addMixingScenario("AVE_75_BX_50ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':50, 'B': (-12,3), 'N': 75})
addMixingScenario("AVE_75_BX_25ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':25, 'B': (-12,3), 'N': 75})
addMixingScenario("AVE_100_BX_50ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':50, 'B': (-12,3), 'N': 100})
addMixingScenario("AVE_100_BX_25ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':25, 'B': (-12,3), 'N': 100})
addMixingScenario("AVE_125_BX_50ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':50, 'B': (-12,3), 'N': 125})
addMixingScenario("AVE_125_BX_25ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':25, 'B': (-12,3), 'N': 125})
addMixingScenario("AVE_150_BX_50ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':50, 'B': (-12,3), 'N': 150})
addMixingScenario("AVE_150_BX_25ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':25, 'B': (-12,3), 'N': 150})
addMixingScenario("AVE_175_BX_50ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':50, 'B': (-12,3), 'N': 175})
addMixingScenario("AVE_175_BX_25ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':25, 'B': (-12,3), 'N': 175})
addMixingScenario("AVE_200_BX_50ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':50, 'B': (-12,3), 'N': 200})
addMixingScenario("AVE_200_BX_25ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':25, 'B': (-12,3), 'N': 200})
addMixingScenario("AVE_140_BX_50ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':50, 'B': (-12,3), 'N': 140})
addMixingScenario("AVE_140_BX_25ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':25, 'B': (-12,3), 'N': 140})


MixingDefaultKey = '2012_Summer_50ns_PoissonOOTPU'
MixingFSDefaultKey = '2012_Summer_inTimeOnly'

def printMe():
    global Mixing
    keys = Mixing.keys()
    keys.sort()
    fskeys=[]
    for key in keys:
        if '_FS' in key:
            fskeys.append(key)
        else:
            print 'addMixingScenario("%s",%s)'%(key,repr(Mixing[key]))

    for key in fskeys:
        print 'addMixingScenario("%s",%s)'%(key,repr(Mixing[key]))


def defineMixing(dict,FS=False):
    commands=[]
    if 'N' in dict:
        if FS:
            commands.append('process.famosPileUp.PileUpSimulator.averageNumber = cms.double(%f)'%(dict['N'],))
        else:
            commands.append('process.mix.input.nbPileupEvents.averageNumber = cms.double(%f)'%(dict['N'],))
        dict.pop('N')
    if 'BX' in dict:
        commands.append('process.mix.bunchspace = cms.int32(%d)'%(dict['BX'],))
        dict.pop('BX')
    if 'B' in dict:
        commands.append('process.mix.minBunch = cms.int32(%d)'%(dict['B'][0],))
        commands.append('process.mix.maxBunch = cms.int32(%d)'%(dict['B'][1],))
        dict.pop('B')
    if 'F' in dict:
        commands.append('process.mix.input.fileNames = cms.untracked.vstring(%s)'%(repr(dict['F'])))
        dict.pop('F')
    return commands
