import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles
corrMETtestInputFiles = pickRelValInputFiles(
    useDAS = True,
    cmsswVersion = 'CMSSW_7_1_0_pre8',
    dataTier = 'GEN-SIM-RECO',
    relVal = 'RelValTTbar_13',
    globalTag = 'PU50ns_PRE_LS171_V10',
    maxVersions = 2
    )

# corrMETtestInputFiles = [
#     '/store/relval/CMSSW_7_1_0_pre8/RelValTTbar_13/GEN-SIM-RECO/PU50ns_PRE_LS171_V10-v1/00000/342381E4-6BE7-E311-AEF0-0025905A60AA.root',
#     ]

##____________________________________________________________________________||
