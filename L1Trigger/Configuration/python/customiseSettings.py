import os.path
import FWCore.ParameterSet.Config as cms
from Configuration.StandardSequences.Eras import eras

def L1TSettingsToCaloStage2Params_v2_1(process):
    process.load("L1Trigger.L1TCalorimeter.caloStage2Params_2016_v2_1_cfi")
    return process

def L1TSettingsToCaloStage2Params_v2_0(process):
    process.load("L1Trigger.L1TCalorimeter.caloStage2Params_2016_v2_0_cfi")
    return process


def L1TSettingsToCaloStage2Params_UserDefine(process):
    print "Loading configuration for calorimeter parameters in user defined file ./caloStage2Params_UserDefine_cfi.py"
    if not (os.path.exists("./caloStage2Params_UserDefine_cfi.py")):
        print "WARNING:"
        print "   Please create file ./caloStage2Params_UserDefine_cfi.py if you want to use the flag: "
        print "   --customise=L1Trigger/Configuration/customiseSettings.L1TSettingsToCaloParams_UserDefine"

    else:
        process.load("./caloStage2Params_UserDefine_cfi")

    return process

