import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles
recoMETtestInputFiles = pickRelValInputFiles(
    useDAS = True,
    cmsswVersion = 'CMSSW_7_1_0_pre2',
    dataTier = 'GEN-SIM-RECO',
    relVal = 'RelValTTbar_13',
    globalTag = 'PU50ns_POSTLS170_V4'
    )

# recoMETtestInputFiles = [
#     '/store/relval/CMSSW_7_1_0_pre2/RelValTTbar_13/GEN-SIM-RECO/PU50ns_POSTLS170_V4-v1/00000/FAA1E1EE-BE8F-E311-B633-0026189438BC.root',
#     ]

##____________________________________________________________________________||
