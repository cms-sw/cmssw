#!/usr/bin/env python3
"""CRAB configuration for one OO2025 DATA dataset.

For multiple datasets, use submit_multidataset_OO_DATA.sh.
"""

from CRABClient.UserUtilities import config

config = config()

config.General.requestName = "HiForest_OO2025_DATA_IonPhysics0_Nebraska_slim_v1"
config.General.workArea = "crab_projects/OO2025_DATA_Nebraska_slim"
config.General.transferOutputs = True
config.General.transferLogs = False

config.JobType.pluginName = "Analysis"
config.JobType.psetName = "forest_miniAOD_run3_DATA.py"
config.JobType.maxMemoryMB = 2500

config.Data.inputDataset = (
    "/IonPhysics0/OORun2025-PromptReco-v1/MINIAOD"
)
config.Data.inputDBS = "global"
config.Data.publication = False
config.Data.splitting = "EventAwareLumiBased"
config.Data.unitsPerJob = 50000
config.Data.totalUnits = -1
config.Data.lumiMask = "Cert_Collisions2025OO_394153_394217_golden.json"

# Change the CERN username and storage site for another account.
config.Data.outLFNDirBase = "/store/user/jvelazqu/HiForest/OO2025/DATA"
config.Data.outputDatasetTag = config.General.requestName

config.Site.storageSite = "T2_US_Nebraska"
config.Site.blacklist = ["T3_UK_ScotGrid_GLA", "T2_DE_DESY"]
