import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles
recoMETtestInputFiles = pickRelValInputFiles(
    useDAS = True,
    cmsswVersion = 'CMSSW_7_1_0_pre4_AK4',
    dataTier = 'GEN-SIM-RECO',
    relVal = 'RelValTTbar_13',
    globalTag = 'PU50ns_POSTLS171_V2',
    maxVersions = 2
    )

# recoMETtestInputFiles = [
#     '/store/relval/CMSSW_7_1_0_pre4_AK4/RelValTTbar_13/GEN-SIM-RECO/PU50ns_POSTLS171_V2-v2/00000/1487CD0E-A4B3-E311-96D2-0025904C678A.root',
#     ]
##____________________________________________________________________________||
