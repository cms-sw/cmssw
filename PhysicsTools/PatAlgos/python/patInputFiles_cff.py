import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

filesRelValZeeGENSIMRECO = cms.untracked.vstring(
    '/store/relval/CMSSW_9_3_0_pre1/RelValZEE_13/GEN-SIM-RECO/PU25ns_92X_upgrade2017_realistic_v7-v1/00000/2A678BC2-3A61-E711-A1F5-0025905A6094.root'
)
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
    '/store/relval/CMSSW_9_2_2/RelValTTbar_13/MINIAODSIM/PU25ns_92X_upgrade2017_realistic_v1-v1/10000/8E7EE25F-294E-E711-A5CC-0025905B8610.root',
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
    '/store/relval/CMSSW_9_2_2/RelValProdTTbar_13/AODSIM/91X_mcRun2_asymptotic_v3-v1/10000/EEB99F74-DA4D-E711-A41C-0025905A48F2.root',
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
    '/store/relval/CMSSW_9_3_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU25ns_92X_upgrade2017_realistic_v7-v1/00000/32EA1438-3D61-E711-8FE7-0025905B85B2.root'
    #'/store/relval/CMSSW_9_2_2/RelValTTbar_13/GEN-SIM-RECO/92X_upgrade2017_realistic_v1-v1/10000/C880B6BD-B14D-E711-A3C5-0CC47A4D767E.root',
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
    '/store/relval/CMSSW_9_2_2/RelValTTbar_13/GEN-SIM-DIGI-RECO/91X_mcRun2_asymptotic_v3_FastSim-v1/10000/8647FB5A-734D-E711-94D8-0025905B85F6.root',
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
    '/store/relval/CMSSW_9_2_2/RelValTTbar_13/GEN-SIM-RECO/PU25ns_92X_upgrade2017_realistic_v1-v1/10000/ECFEA1BD-BF4D-E711-A404-0CC47A7C345C.root',
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
