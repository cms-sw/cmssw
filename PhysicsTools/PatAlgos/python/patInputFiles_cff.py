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
        '/store/relval/CMSSW_8_0_0/RelValTTbar_13/MINIAODSIM/PU25ns_80X_mcRun2_asymptotic_v4-v1/10000/A65CD249-BFDA-E511-813A-0025905A6066.root'
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
        '/store/relval/CMSSW_8_0_0/RelValProdTTbar_13/AODSIM/80X_mcRun2_asymptotic_v4-v1/10000/DE81ABBF-1DDA-E511-8AF8-0026189438B5.root'
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
        '/store/relval/CMSSW_8_0_0/RelValTTbar_13/GEN-SIM-RECO/80X_mcRun2_asymptotic_v4-v1/10000/1C687FB0-7BD9-E511-AFED-0CC47A78A4BA.root'
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
        '/store/relval/CMSSW_8_0_0/RelValTTbar_13/GEN-SIM-DIGI-RECO/PU25ns_80X_mcRun2_asymptotic_v4_FastSim-v2/10000/00AADAB6-63DD-E511-8C34-002618943953.root'
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
        '/store/relval/CMSSW_8_0_0/RelValTTbar_13/GEN-SIM-RECO/PU25ns_80X_mcRun2_asymptotic_v4-v1/10000/42D6DF66-9DDA-E511-9200-0CC47A4D7670.root'
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
        '/store/relval/CMSSW_8_0_0/SingleMu/MINIAOD/80X_dataRun2_v5_RelVal_mu2012D-v3/10000/06A44F40-ECDD-E511-89D7-0CC47A78A3D8.root'
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
