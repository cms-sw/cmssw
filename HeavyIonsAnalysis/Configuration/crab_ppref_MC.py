#!/usr/bin/env python3
"""CRAB configuration for one ppref2024 MC dataset.
"""

from CRABClient.UserUtilities import config

config = config()

config.General.requestName = "HiForest_ppref2024_MC_Pythia8_Nebraska_v3"
config.General.workArea = "crab_projects/ppref2024_MC_Nebraska"
config.General.transferOutputs = True
config.General.transferLogs = False

config.JobType.pluginName = "Analysis"
config.JobType.psetName = "forest_miniAOD_run3_ppref_MC.py"
config.JobType.maxMemoryMB = 2500

config.Data.inputDataset = (
    "/QCD_pThat-15to1200_TuneCP5_5p36TeV_pythia8/RunIIIpp5p36Winter24MiniAOD-141X_mcRun3_2024_realistic_ppRef5TeV_v7-v2/MINIAODSIM"
)
config.Data.inputDBS = "global"
config.Data.publication = False
config.Data.splitting = "EventAwareLumiBased"
config.Data.unitsPerJob = 100000
config.Data.totalUnits = -1

# Change the CERN username and storage site for another account.
config.Data.outLFNDirBase = "/store/user/jvelazqu/HiForest/ppref2024/MC"
config.Data.outputDatasetTag = config.General.requestName

config.Site.storageSite = "T2_US_Nebraska"
