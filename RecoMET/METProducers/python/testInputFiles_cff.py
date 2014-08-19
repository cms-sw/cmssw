import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles
recoMETtestInputFiles = pickRelValInputFiles(
    useDAS = True,
    cmsswVersion = 'CMSSW_7_2_0_pre1',
    dataTier = 'GEN-SIM-RECO',
    relVal = 'RelValTTbar_13',
    globalTag = 'PU50ns_POSTLS172_V2',
    maxVersions = 2
    )

# recoMETtestInputFiles = [
#     '/store/relval/CMSSW_7_2_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_POSTLS172_V2-v1/00000/0AA51FF6-8EFD-E311-B591-0025905A6068.root',
#     ]
##____________________________________________________________________________||
