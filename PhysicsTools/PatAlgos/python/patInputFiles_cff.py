import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

# /RelValProdTTbar/CMSSW_7_0_0_pre11-START70_V4-v1/AODSIM
filesRelValProdTTbarAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_0_0_pre11'
                        #, relVal        = 'RelValProdTTbar'
                        #, globalTag     = 'START70_V4'
                        #, dataTier      = 'AODSIM'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_0_0/RelValProdTTbar/AODSIM/START70_V6-v2/00000/A66126BC-C998-E311-961B-003048FEB916.root'
    )

# /RelValProdTTbar/CMSSW_7_0_0_pre11-START70_V4-v1/GEN-SIM-RECO
filesRelValProdTTbarGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_0_0_pre11'
                        #, relVal        = 'RelValProdTTbar'
                        #, globalTag     = 'START70_V4'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_0_0/RelValProdTTbar/GEN-SIM-RECO/START70_V6-v2/00000/103447B4-BF98-E311-B01D-02163E0079BC.root'
    )

# /RelValTTbar_13/CMSSW_7_0_0_pre11-PU50ns_POSTLS162_V5-v1/GEN-SIM-RECO
filesRelValTTbarPileUpGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_0_0_pre11'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU50ns_POSTLS162_V5'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_0_0/RelValTTbar_13/GEN-SIM-RECO/PU50ns_POSTLS170_V4-v2/00000/265B9219-FF98-E311-BF4A-02163E00EA95.root'
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
        '/store/relval/CMSSW_7_0_0/SingleMu/RECO/GR_R_70_V1_RelVal_zMu2012D-v2/00000/0259E46E-F698-E311-8CFD-003048FF9AC6.root'
    )
