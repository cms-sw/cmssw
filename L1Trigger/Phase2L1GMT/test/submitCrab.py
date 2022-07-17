import os


data = {
#    'DY0':'/DYToLL_M-50_TuneCP5_14TeV-pythia8/Phase2HLTTDRWinter20DIGI-NoPU_pilot_110X_mcRun4_realistic_v3-v2/GEN-SIM-DIGI-RAW',
#    'DY140a':'/DYToLL_M-50_TuneCP5_14TeV-pythia8/Phase2HLTTDRWinter20DIGI-PU140_pilot1_110X_mcRun4_realistic_v3-v2/GEN-SIM-DIGI-RAW',
#    'DY140b':'/DYToLL_M-50_TuneCP5_14TeV-pythia8/Phase2HLTTDRWinter20DIGI-PU140_pilot_110X_mcRun4_realistic_v3-v2/GEN-SIM-DIGI-RAW',
    'DY200a':'/DYToLL_M-50_TuneCP5_14TeV-pythia8/Phase2HLTTDRWinter20DIGI-PU200_pilot_110X_mcRun4_realistic_v3-v2/GEN-SIM-DIGI-RAW',
    'DY200b':'/DYToLL_M-50_TuneCP5_14TeV-pythia8/Phase2HLTTDRWinter20DIGI-PU200_pilot2_110X_mcRun4_realistic_v3-v2/GEN-SIM-DIGI-RAW',
#    'SingleMu0':'/DoubleMuon_gun_FlatPt-1To100/Phase2HLTTDRWinter20DIGI-NoPU_110X_mcRun4_realistic_v3-v2/GEN-SIM-DIGI-RAW',
#    'SingleMu140':'/DoubleMuon_gun_FlatPt-1To100/Phase2HLTTDRWinter20DIGI-PU140_110X_mcRun4_realistic_v3-v2/GEN-SIM-DIGI-RAW',
#    'SingleMu200':'/DoubleMuon_gun_FlatPt-1To100/Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v3/GEN-SIM-DIGI-RAW',
#    'TT200':'/TT_TuneCP5_14TeV-powheg-pythia8/Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v2/GEN-SIM-DIGI-RAW',
#    'JPsiToMuMu':'/JPsiToMuMu_Pt0to100-pythia8_TuneCP5-gun/Phase2HLTTDRWinter20DIGI-NoPU_110X_mcRun4_realistic_v3-v2/GEN-SIM-DIGI-RAW',
    'JPsiToMuMu200a':'/JPsiToMuMu_Pt0to100-pythia8_TuneCP5-gun/Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v2/GEN-SIM-DIGI-RAW',
    'JPsiToMuMu200b':'/JPsiToMuMu_Pt0to100-pythia8_TuneCP5-gun/Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3_ext1-v3/GEN-SIM-DIGI-RAW',
    'MinBias200_a':'/MinBias_TuneCP5_14TeV-pythia8/Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v3/GEN-SIM-DIGI-RAW',
    'MinBias200_b':'/MinBias_TuneCP5_14TeV-pythia8/Phase2HLTTDRWinter20DIGI-PU200_BSzpz35_BSzpz35_110X_mcRun4_realistic_v3_ext1-v2/GEN-SIM-DIGI-RAW',
    'MinBias200_c':'/MinBias_TuneCP5_14TeV-pythia8/Phase2HLTTDRWinter20DIGI-PU200_withNewMB_110X_mcRun4_realistic_v3_ext1-v2/GEN-SIM-DIGI-RAW'
}


for tag,dataset in  data.iteritems():
    FILE="""
from CRABClient.UserUtilities import config
config = config()
config.General.requestName = 'skim_{tag}'
config.General.workArea = 'crab_projects'
config.General.transferOutputs = True
config.General.transferLogs = False
config.JobType.pluginName = 'Analysis'
config.JobType.psetName = 'runGMT.py'
config.Data.inputDataset = '{dataset}'
config.Data.inputDBS = 'global'
config.Data.splitting = 'FileBased'
config.Data.unitsPerJob = 1
config.Data.outLFNDirBase = '/store/user/bachtis/L1TF4'
config.Data.publication = True
config.Data.ignoreLocality = True
config.Data.outputDatasetTag = 'PHASEII_{tag}'
config.Site.storageSite = 'T3_US_FNALLPC'
config.Site.whitelist = ['T2_US_*']
config.JobType.allowUndistributedCMSSW = True
config.JobType.maxMemoryMB = 4000
""".format(tag=tag,dataset=dataset)
    f=open("crab_{tag}.py".format(tag=tag),"w")
    print(FILE)
    f.write(FILE)
    f.close()
    os.system("crab submit -c crab_{PT}.py".format(PT=tag))


