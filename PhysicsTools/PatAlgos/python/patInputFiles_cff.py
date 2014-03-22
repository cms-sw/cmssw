import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

# /RelValProdTTbar/CMSSW_7_1_0_pre1-START70_V5-v1/AODSIM
filesRelValProdTTbarAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_1_0_pre1'
                        #, relVal        = 'RelValProdTTbar'
                        #, globalTag     = 'START70_V5'
                        #, dataTier      = 'AODSIM'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_1_0_pre1/RelValProdTTbar/AODSIM/START70_V5-v1/00000/F4EB9159-3286-E311-8D80-02163E008DB4.root'
    )

# /RelValProdTTbar/CMSSW_7_1_0_pre1-START70_V5-v1/GEN-SIM-RECO
filesRelValProdTTbarGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_1_0_pre1'
                        #, relVal        = 'RelValProdTTbar'
                        #, globalTag     = 'START70_V5'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_1_0_pre1/RelValProdTTbar/GEN-SIM-RECO/START70_V5-v1/00000/14842A6B-2086-E311-B5CB-02163E00E8DA.root'
    )

# /RelValTTbar/CMSSW_7_1_0_pre1-PU_START70_V5_FastSim-v2/GEN-SIM-DIGI-RECO
filesRelValTTbarPileUpFastSimGENSIMDIGIRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_1_0_pre1'
                        #, relVal        = 'RelValTTbar'
                        #, globalTag     = 'PU_START70_V5_FastSim'
                        #, dataTier      = 'GEN-SIM-DIGI-RECO'
                        #, maxVersions   = 2
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_1_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/PU_START70_V5_FastSim-v2/00000/00865693-258F-E311-B41F-0025905A6092.root'
    )

# /RelValTTbar_13/CMSSW_7_1_0_pre1-PU50ns_POSTLS170_V1-v1/GEN-SIM-RECO
filesRelValTTbarPileUpGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_1_0_pre1'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU50ns_POSTLS170_V1'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_1_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_POSTLS170_V1-v1/00000/0E08B589-BC8B-E311-BD48-02163E00EA0D.root'
    )

# /SingleMu/CMSSW_6_2_0_pre8-PRE_62_V8_RelVal_mu2012D-v1/RECO
filesSingleMuRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_6_2_0_pre8' # no 70X data RelVals at CERN
                        #, relVal        = 'SingleMu'
                        #, dataTier      = 'RECO'
                        #, globalTag     = 'PRE_62_V8_RelVal_mu2012D'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_6_2_0_pre8/SingleMu/RECO/PRE_62_V8_RelVal_mu2012D-v1/00000/005835E9-05E0-E211-BA7B-003048F1C7C0.root'
    )
