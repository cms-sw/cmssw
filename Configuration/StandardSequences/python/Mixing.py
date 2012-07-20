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
addMixingScenario("E8TeV_AVE_10_BX_50ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX':50, 'B': (-3,2), 'N': 10})
addMixingScenario("E8TeV_AVE_16_BX_25ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi'})
addMixingScenario("E8TeV_AVE_16_BX_50ns",{'file': 'SimGeneral.MixingModule.mix_E8TeV_AVE_16_BX_25ns_cfi','BX': 50, 'B': (-3,2)})
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
addMixingScenario("ProdStep2",{'file': 'SimGeneral.MixingModule.mixProdStep2_cfi'})

##fastsim section
#addMixingScenario("FS_NoPileUp",{'file': 'FastSimulation.PileUpProducer.PileUpSimulator_cff', 'N': 0})
#addMixingScenario("FS_156BxLumiPileUp",{'file': 'FastSimulation.PileUpProducer.PileUpSimulator_cff', 'N': 2})
#addMixingScenario("FS_HighLumiPileUp",{'file': 'FastSimulation.PileUpProducer.PileUpSimulator_cff', 'N': 20})
#addMixingScenario("FS_InitialPileUp",{'file': 'FastSimulation.PileUpProducer.PileUpSimulator_cff', 'N': 3.8})
#addMixingScenario("FS_LowLumiPileUp",{'file': 'FastSimulation.PileUpProducer.PileUpSimulator_cff', 'N': 7.1})
addMixingScenario("FS_NoPileUp",{'file': 'FastSimulation.PileUpProducer.PileUpSimulator_NoPileUp_cff'})
addMixingScenario("FS_156BxLumiPileUp",{'file': 'FastSimulation.PileUpProducer.PileUpSimulator_156BxLumiPileUp_cff'})
addMixingScenario("FS_HighLumiPileUp",{'file': 'FastSimulation.PileUpProducer.PileUpSimulator_HighLumiPileUp_cff'})
addMixingScenario("FS_InitialPileUp",{'file': 'FastSimulation.PileUpProducer.PileUpSimulator_InitialPileUp_cff'})
addMixingScenario("FS_LowLumiPileUp",{'file': 'FastSimulation.PileUpProducer.PileUpSimulator_LowLumiPileUp_cff'})
addMixingScenario("FS_FlatDist10_2011EarlyData_50ns",{'file': 'FastSimulation.PileUpProducer.PileUpSimulator_FlatDist10_2011EarlyData_50ns_cff'})

##slch section
addMixingScenario("SLHC_LowLumiPileUp_Phase1_R30F12",{'file':'SLHCUpgradeSimulations.Geometry.mixLowLumPU_Phase1_R30F12_cff'})
addMixingScenario("SLHC_LowLumiPileUp_Phase1_R30F12_smpx",{'file':'SLHCUpgradeSimulations.Geometry.mixLowLumPU_Phase1_R30F12_smpx_cff'})
addMixingScenario("SLHC_LowLumiPileUp_Phase1_R34F16",{'file':'SLHCUpgradeSimulations.Geometry.mixLowLumPU_Phase1_R34F16_cff'})
addMixingScenario("SLHC_LowLumiPileUp_Phase1_R34F16_smpx",{'file':'SLHCUpgradeSimulations.Geometry.mixLowLumPU_Phase1_R34F16_smpx_cff'})
addMixingScenario("SLHC_LowLumiPileUp_stdgeom",{'file':'SLHCUpgradeSimulations.Geometry.mixLowLumPU_stdgeom_cff'})
addMixingScenario("SLHC_LowLumiPileUp_Phase1_R30F12_HCal",{'file':'SLHCUpgradeSimulations.Geometry.mixLowLumPU_Phase1_R30F12_HCal_cff'})
addMixingScenario("SLHC_LowLumiPileUp_Phase1_R34F16_HCal",{'file':'SLHCUpgradeSimulations.Geometry.mixLowLumPU_Phase1_R34F16_HCal_cff'})

MixingDefaultKey = 'NoPileUp'

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
