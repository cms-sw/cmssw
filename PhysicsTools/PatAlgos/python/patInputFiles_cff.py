import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

# /RelValTTbar_13/CMSSW_8_0_0-PU25ns_80X_mcRun2_asymptotic_v4-v1/MINIAODSIM
filesRelValTTbarPileUpMINIAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_8_0_0'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU25ns_80X_mcRun2_asymptotic_v4'
                        #, dataTier      = 'MINIAODSIM'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_9_1_0_pre2/RelValTTbar_13/MINIAODSIM/90X_upgrade2017_realistic_v20-v2/00000/16132980-3019-E711-AD34-0025905A6110.root'
    )

# /RelValProdTTbar_13/CMSSW_8_0_0-80X_mcRun2_asymptotic_v4-v1/AODSIM
filesRelValProdTTbarAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_8_0_0'
                        #, relVal        = 'RelValProdTTbar_13'
                        #, globalTag     = '80X_mcRun2_asymptotic_v4'
                        #, dataTier      = 'AODSIM'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_9_1_0_pre2/RelValTTbar_13/GEN-SIM-RECO/90X_upgrade2017_realistic_v20-v2/00000/2257937F-3019-E711-BF48-0CC47A4D7678.root'
    )

# /RelValTTbar_13/CMSSW_8_0_0-80X_mcRun2_asymptotic_v4-v1/GEN-SIM-RECO
filesRelValTTbarGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_8_0_0'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = '80X_mcRun2_asymptotic_v4'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_9_1_0_pre2/RelValTTbar_13/GEN-SIM-RECO/90X_upgrade2017_realistic_v20-v2/00000/2257937F-3019-E711-BF48-0CC47A4D7678.root'
    )

# /RelValTTbar_13/CMSSW_8_0_0-PU25ns_80X_mcRun2_asymptotic_v4_FastSim-v2/GEN-SIM-DIGI-RECO
filesRelValTTbarPileUpFastSimGENSIMDIGIRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_8_0_0'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU25ns_80X_mcRun2_asymptotic_v4_FastSim'
                        #, dataTier      = 'GEN-SIM-DIGI-RECO'
                        #, maxVersions   = 2
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_9_1_0_pre2/RelValTTbar_13/GEN-SIM-DIGI-RECO/PU25ns_90X_mcRun2_asymptotic_v5_FastSim-v1/00000/0039AD51-331A-E711-863C-0025905A60B4.root',
    )

# /RelValTTbar_13/CMSSW_8_0_0-PU25ns_80X_mcRun2_asymptotic_v4-v1/GEN-SIM-RECO
filesRelValTTbarPileUpGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_8_0_0'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU25ns_80X_mcRun2_asymptotic_v4'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_9_1_0_pre2/RelValTTbar_13/GEN-SIM-RECO/PU25ns_90X_mcRun2_asymptotic_v5-v1/00000/E27EA6EC-2B1A-E711-84EC-0CC47A4D75EC.root',
    )

# /SingleMu/CMSSW_8_0_0-80X_dataRun2_v5_RelVal_mu2012D-v3/MINIAOD
filesRelValSingleMuMINIAOD = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_8_0_0'
                        #, relVal        = 'SingleMu'
                        #, globalTag     = '80X_dataRun2_v5_RelVal_mu2012D'
                        #, dataTier      = 'MINIAOD'
                        #, maxVersions   = 3
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_9_1_0_pre2/SingleMuon/MINIAOD/90X_dataRun2_relval_v6_RelVal_sigMu2016E-v1/00000/96231232-361A-E711-96B5-0CC47A7C3430.root'
    )

# /SingleMu/CMSSW_8_0_0-80X_dataRun2_v5_RelVal_mu2012D-v3/RECO
filesSingleMuRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_8_0_0'
                        #, relVal        = 'SingleMu'
                        #, dataTier      = 'RECO'
                        #, globalTag     = '80X_dataRun2_v5_RelVal_mu2012D'
                        #, maxVersions   = 3
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_8_0_0/SingleMu/RECO/80X_dataRun2_v5_RelVal_mu2012D-v3/10000/002B5DB2-28DD-E511-BEFD-0CC47A4D7638.root'
    )
