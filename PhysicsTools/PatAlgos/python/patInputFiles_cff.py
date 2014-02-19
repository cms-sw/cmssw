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
        '/store/relval/CMSSW_7_0_0_pre11/RelValProdTTbar/AODSIM/START70_V4-v1/00000/D0516C65-766A-E311-B744-00259059642E.root'
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
        '/store/relval/CMSSW_7_0_0_pre11/RelValProdTTbar/GEN-SIM-RECO/START70_V4-v1/00000/0EA82C3C-646A-E311-9CB3-0025905A6070.root'
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
        '/store/relval/CMSSW_7_0_0_pre11/RelValTTbar_13/GEN-SIM-RECO/PU50ns_POSTLS162_V5-v1/00000/08DFDC0E-796A-E311-8912-5404A63886EC.root'
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
