import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

# /RelValTTbar_13/CMSSW_7_6_0_pre4-PU25ns_76X_mcRun2_asymptotic_v1-v1/MINIAODSIM
filesRelValTTbarPileUpMINIAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_6_0_pre4'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU25ns_76X_mcRun2_asymptotic_v1'
                        #, dataTier      = 'MINIAODSIM'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_6_0_pre4/RelValTTbar_13/MINIAODSIM/PU25ns_76X_mcRun2_asymptotic_v1-v1/00000/C63FAB0C-ED4F-E511-A453-0025905A7786.root'
    )

# /RelValProdTTbar_13/CMSSW_7_6_0_pre4-76X_mcRun2_asymptotic_v1-v1/AODSIM
filesRelValProdTTbarAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_6_0_pre4'
                        #, relVal        = 'RelValProdTTbar_13'
                        #, globalTag     = '76X_mcRun2_asymptotic_v1'
                        #, dataTier      = 'AODSIM'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_6_0_pre4/RelValProdTTbar_13/AODSIM/76X_mcRun2_asymptotic_v1-v1/00000/90C5FA7A-7B4F-E511-BC27-002618943916.root'
    )

# /RelValTTbar_13/CMSSW_7_6_0_pre4-76X_mcRun2_asymptotic_v1-v1/GEN-SIM-RECO
filesRelValTTbarGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_6_0_pre4'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = '76X_mcRun2_asymptotic_v1'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_6_0_pre4/RelValTTbar_13/GEN-SIM-RECO/76X_mcRun2_asymptotic_v1-v1/00000/24C33B07-5E4F-E511-B11F-0025905A6094.root'
    )

# /RelValTTbar_13/CMSSW_7_6_0_pre4-PU25ns_76X_mcRun2_asymptotic_v1_FastSim-v1/GEN-SIM-DIGI-RECO
filesRelValTTbarPileUpFastSimGENSIMDIGIRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_6_0_pre4'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU25ns_76X_mcRun2_asymptotic_v1_FastSim'
                        #, dataTier      = 'GEN-SIM-DIGI-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        #'/store/relval/CMSSW_7_6_0_pre4/RelValTTbar_13/GEN-SIM-DIGI-RECO/PU25ns_76X_mcRun2_asymptotic_v1_FastSim-v1/00000/0026DA13-BF4F-E511-A342-0025905A6088.root'
    # temporary: produce fastsim sample on the fly
    # can be restored as soon as relval samples are available with the new fastsim rechits
    "file:ttbarForFastSimTest.root"
    )

# /RelValTTbar_13/CMSSW_7_6_0_pre4-PU25ns_76X_mcRun2_asymptotic_v1-v1/GEN-SIM-RECO
filesRelValTTbarPileUpGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_6_0_pre4'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU25ns_76X_mcRun2_asymptotic_v1'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_6_0_pre4/RelValTTbar_13/GEN-SIM-RECO/PU25ns_76X_mcRun2_asymptotic_v1-v1/00000/24C18DA0-6A4F-E511-AE07-0025905A48B2.root'
    )

# /SingleMu/CMSSW_7_6_0_pre4-75X_dataRun1_HLT_frozen_v2_RelVal_mu2012D-v1/MINIAOD
filesRelValSingleMuMINIAOD = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_6_0_pre4'
                        #, relVal        = 'SingleMu'
                        #, globalTag     = '75X_dataRun1_HLT_frozen_v2_RelVal_mu2012D'
                        #, dataTier      = 'MINIAOD'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_6_0_pre4/SingleMu/MINIAOD/75X_dataRun1_HLT_frozen_v2_RelVal_mu2012D-v1/00000/0A9FB3D9-A550-E511-93C5-0025905B85AE.root'
    )

# /SingleMu/CMSSW_7_6_0_pre4-75X_dataRun1_HLT_frozen_v2_RelVal_mu2012D-v1/RECO
filesSingleMuRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_6_0_pre4'
                        #, relVal        = 'SingleMu'
                        #, dataTier      = 'RECO'
                        #, globalTag     = '75X_dataRun1_HLT_frozen_v2_RelVal_mu2012D'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_6_0_pre4/SingleMu/RECO/75X_dataRun1_HLT_frozen_v2_RelVal_mu2012D-v1/00000/004C77E9-7050-E511-B458-0025905B8590.root'
    )
