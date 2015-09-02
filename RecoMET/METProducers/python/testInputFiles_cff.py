import FWCore.ParameterSet.Config as cms

##____________________________________________________________________________||
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles
# temporary: produce fastsim sample on the fly
# can be restored as soon as relval samples are available with the new fastsim rechits
#recoMETtestInputFiles = pickRelValInputFiles(
#    useDAS = True,
#    cmsswVersion = 'CMSSW_7_6_0_pre3',
#    dataTier = 'GEN-SIM-DIGI-RECO',
#    relVal = 'RelValTTbar_13',
#    globalTag = '75X_mcRun2_asymptotic_v2_FastSim',
#    maxVersions = 2
#    )
recoMETtestInputFiles = "file:ttbarForMetTests.root"

# recoMETtestInputFiles = [
#     '/store/relval/CMSSW_7_2_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_POSTLS172_V2-v1/00000/0AA51FF6-8EFD-E311-B591-0025905A6068.root',
#     ]
##____________________________________________________________________________||
