import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

# /RelValTTbar_13/CMSSW_7_2_2_patch1-PU50ns_MCRUN2_72_V0-v1/MINIAODSIM
filesRelValProdTTbarPileUpMINIAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_2_2_patch1'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU50ns_MCRUN2_72_V0'
                        #, dataTier      = 'MINIAODSIM'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_2_2_patch1/RelValTTbar_13/MINIAODSIM/PU50ns_MCRUN2_72_V0-v1/00000/0683685E-8D73-E411-9A7A-0025905A60D0.root'
    )

# /RelValProdTTbar_13/CMSSW_7_2_2_patch1-MCRUN2_72_V1-v1/AODSIM
filesRelValProdTTbarAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_2_2_patch1'
                        #, relVal        = 'RelValProdTTbar_13'
                        #, globalTag     = 'MCRUN2_72_V1'
                        #, dataTier      = 'AODSIM'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_2_2_patch1/RelValProdTTbar_13/AODSIM/MCRUN2_72_V1-v1/00000/3C63AAFD-A973-E411-A1CC-0025905A60BC.root'
    )

# /RelValProdTTbar_13/CMSSW_7_2_2_patch1-MCRUN2_72_V1-v1/GEN-SIM-RECO
filesRelValProdTTbarGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_2_2_patch1'
                        #, relVal        = 'RelValProdTTbar_13'
                        #, globalTag     = 'MCRUN2_72_V1'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_2_2_patch1/RelValProdTTbar_13/GEN-SIM-RECO/MCRUN2_72_V1-v1/00000/583E14D9-A473-E411-BC5B-0025905A60A8.root'
    )

# /RelValTTbar/CMSSW_7_2_0_pre7-PU_PRE_STA72_V4_FastSim-v1/GEN-SIM-DIGI-RECO
filesRelValTTbarPileUpFastSimGENSIMDIGIRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_2_0_pre7'
                        #, relVal        = 'RelValTTbar'
                        #, globalTag     = 'PU_PRE_STA72_V4_FastSim'
                        #, dataTier      = 'GEN-SIM-DIGI-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_2_0_pre7/RelValTTbar/GEN-SIM-DIGI-RECO/PU_PRE_STA72_V4_FastSim-v1/00000/009B474D-1C4B-E411-8347-0026189438EF.root'
    )

# /RelValTTbar_13/CMSSW_7_2_2_patch1-PU50ns_MCRUN2_72_V0-v1/GEN-SIM-RECO
filesRelValTTbarPileUpGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_2_2_patch1'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU50ns_MCRUN2_72_V0'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_2_2_patch1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_MCRUN2_72_V0-v1/00000/580E5F72-7D73-E411-B8F7-0025905B85E8.root'
    )

# /SingleMu/CMSSW_7_2_2_patch1-GR_R_72_V12A_RelVal_mu2012D-v1/MINIAOD
filesRelValSingleMuMINIAOD = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_2_2_patch1'
                        #, relVal        = 'SingleMu'
                        #, globalTag     = 'GR_R_72_V12A_RelVal_mu2012D'
                        #, dataTier      = 'MINIAOD'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_2_2_patch1/SingleMu/MINIAOD/GR_R_72_V12A_RelVal_mu2012D-v1/00000/0A4421EB-F573-E411-9579-0025905A6134.root'
    )

# /SingleMu/CMSSW_7_2_2_patch1-GR_R_72_V12A_RelVal_mu2012D-v1/RECO # not at CERN
filesSingleMuRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_2_2_patch1'
                        #, relVal        = 'SingleMu'
                        #, dataTier      = 'RECO'
                        #, globalTag     = 'GR_R_72_V12A_RelVal_mu2012D'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_2_2_patch1/SingleMu/RECO/GR_R_72_V12A_RelVal_mu2012D-v1/00000/000A27E6-7573-E411-BB60-0025905A60D0.root'
    )
