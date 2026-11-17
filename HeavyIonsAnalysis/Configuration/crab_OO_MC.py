#!/usr/bin/env python3
"""CRAB configuration for one OO2025 MC dataset.
"""

from CRABClient.UserUtilities import config

config = config()

config.General.requestName = "HiForest_OO2025_MC_Pythia8_Nebraska_slim_v1"
config.General.workArea = "crab_projects/OO2025_MC_Nebraska_slim"
config.General.transferOutputs = True
config.General.transferLogs = False

config.JobType.pluginName = "Analysis"
config.JobType.psetName = "forest_miniAOD_run3_MC.py"
config.JobType.maxMemoryMB = 2500

config.Data.inputDataset = (
    "/QCD-dijet_Pthat-15_TuneCP5_OO_5p36TeV_pythia8/HINOOSpring25MiniAOD-150X_mcRun3_2025_forOO_realistic_v7-v2/MINIAODSIM"
)
config.Data.inputDBS = "global"
config.Data.publication = False
config.Data.splitting = "EventAwareLumiBased"
config.Data.unitsPerJob = 30000
config.Data.totalUnits = -1

# Change the CERN username and storage site for another account.
config.Data.outLFNDirBase = "/store/user/jvelazqu/HiForest/OO2025/MC"
config.Data.outputDatasetTag = config.General.requestName

config.Site.storageSite = "T2_US_Nebraska"
config.Site.blacklist = ["T3_UK_ScotGrid_GLA", "T2_DE_DESY"]
