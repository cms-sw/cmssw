#!/usr/bin/env python3
"""CRAB configuration for one ppref2024 DATA dataset.

For multiple datasets, use submit_multidataset_ppref_DATA.sh.
"""

from CRABClient.UserUtilities import config

config = config()

config.General.requestName = "HiForest_ppref2024_DATA_PPRefHardProbes0_v1"
config.General.workArea = "crab_projects/ppref2024_DATA"
config.General.transferOutputs = True
config.General.transferLogs = False

config.JobType.pluginName = "Analysis"
config.JobType.psetName = "forest_miniAOD_run3_ppref_DATA.py"
config.JobType.maxMemoryMB = 3000

config.Data.inputDataset = (
    "/PPRefHardProbes0/Run2024J-PromptReco-v1/MINIAOD"
)
config.Data.inputDBS = "global"
config.Data.publication = False
config.Data.splitting = "EventAwareLumiBased"
config.Data.unitsPerJob = 50000
config.Data.totalUnits = -1
config.Data.lumiMask = "Cert_Collisions2024_ppref_387474_387721_golden.json"

# Change the CERN username and storage site for another account.
config.Data.outLFNDirBase = "/store/user/jvelazqu/HiForest/ppref2024/DATA"
config.Data.outputDatasetTag = config.General.requestName

config.Site.storageSite = "T2_US_Nebraska"
config.Site.blacklist = ["T3_UK_ScotGrid_GLA", "T2_DE_DESY"]
