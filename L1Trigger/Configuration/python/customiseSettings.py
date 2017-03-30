import os.path
import FWCore.ParameterSet.Config as cms

def L1TSettingsToCaloStage2Params_2017_v1_0_HI_inconsistent(process):
    process.load("L1Trigger.L1TCalorimeter.caloStage2Params_2017_v1_0_HI_inconsistent_cfi")
    return process

def L1TSettingsToCaloStage2Params_2017_v1_0_inconsistent(process):
    process.load("L1Trigger.L1TCalorimeter.caloStage2Params_2017_v1_0_inconsistent_cfi")
    return process

def L1TSettingsToCaloStage2Params_v3_3_1_HI(process):
    process.load("L1Trigger.L1TCalorimeter.caloStage2Params_2016_v3_3_1_HI_cfi")
    return process

def L1TSettingsToCaloStage2Params_v3_3_1(process):
    process.load("L1Trigger.L1TCalorimeter.caloStage2Params_2016_v3_3_1_cfi")
    return process

def L1TSettingsToCaloStage2Params_v3_3_HI(process):
    process.load("L1Trigger.L1TCalorimeter.caloStage2Params_2016_v3_3_HI_cfi")
    return process

def L1TSettingsToCaloStage2Params_v3_3(process):
    process.load("L1Trigger.L1TCalorimeter.caloStage2Params_2016_v3_3_cfi")
    return process

def L1TSettingsToCaloStage2Params_v3_2(process):
    process.load("L1Trigger.L1TCalorimeter.caloStage2Params_2016_v3_2_cfi")
    return process

def L1TSettingsToCaloStage2Params_v3_1(process):
    process.load("L1Trigger.L1TCalorimeter.caloStage2Params_2016_v3_1_cfi")
    return process

def L1TSettingsToCaloStage2Params_v3_0(process):
    process.load("L1Trigger.L1TCalorimeter.caloStage2Params_2016_v3_0_cfi")
    return process

def L1TSettingsToCaloStage2Params_v2_2(process):
    process.load("L1Trigger.L1TCalorimeter.caloStage2Params_2016_v2_2_cfi")
    return process

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

