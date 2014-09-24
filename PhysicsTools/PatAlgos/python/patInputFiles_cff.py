import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

# /RelValProdTTbar/CMSSW_7_0_0-START70_V6-v2/AODSIM
filesRelValProdTTbarAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_0_0'
                        #, relVal        = 'RelValProdTTbar'
                        #, globalTag     = 'START70_V6'
                        #, dataTier      = 'AODSIM'
                        #, maxVersions   = 2
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_0_0/RelValProdTTbar/AODSIM/START70_V6-v2/00000/A66126BC-C998-E311-961B-003048FEB916.root'
    )

# /RelValProdTTbar/CMSSW_7_0_0-START70_V6-v2/GEN-SIM-RECO
filesRelValProdTTbarGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_0_0'
                        #, relVal        = 'RelValProdTTbar'
                        #, globalTag     = 'START70_V6'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 2
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_0_0/RelValProdTTbar/GEN-SIM-RECO/START70_V6-v2/00000/103447B4-BF98-E311-B01D-02163E0079BC.root'
    )

# /RelValTTbar/CMSSW_7_0_0-PU_START70_V6_FastSim-v2/GEN-SIM-DIGI-RECO
filesRelValTTbarPileUpFastSimGENSIMDIGIRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_0_0'
                        #, relVal        = 'RelValTTbar'
                        #, globalTag     = 'PU_START70_V6_FastSim'
                        #, dataTier      = 'GEN-SIM-DIGI-RECO'
                        #, maxVersions   = 2
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_0_0/RelValTTbar/GEN-SIM-DIGI-RECO/PU_START70_V6_FastSim-v2/00000/0052DFFB-B398-E311-BE47-003048FE9B78.root'
    )

# /RelValTTbar_13/CMSSW_7_0_0-PU50ns_POSTLS170_V4-v2/GEN-SIM-RECO
filesRelValTTbarPileUpGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_0_0'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU50ns_POSTLS170_V4'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 2
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_0_0/RelValTTbar_13/GEN-SIM-RECO/PU50ns_POSTLS170_V4-v2/00000/265B9219-FF98-E311-BF4A-02163E00EA95.root'
    )

# /SingleMu/CMSSW_7_0_0-GR_R_70_V1_RelVal_mu2012D-v2/RECO
filesSingleMuRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_0_0' # not at CERN
                        #, relVal        = 'SingleMu'
                        #, dataTier      = 'RECO'
                        #, globalTag     = 'GR_R_70_V1_RelVal_mu2012D'
                        #, maxVersions   = 2
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_0_0/SingleMu/RECO/GR_R_70_V1_RelVal_mu2012D-v2/00000/00F93EBE-A498-E311-B2A7-002590596490.root'
    )
