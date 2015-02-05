import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

# /RelValTTbar_13/CMSSW_7_4_0_pre5-PU25ns_MCRUN2_73_V7-v1/MINIAODSIM
filesRelValProdTTbarPileUpMINIAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_4_0_pre5'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU25ns_MCRUN2_73_V7'
                        #, dataTier      = 'MINIAODSIM'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/MINIAODSIM/PU25ns_MCRUN2_74_V1-v3/00000/28C3D04B-B6AB-E411-A0A3-0025905A613C.root'
    )

# /RelValProdTTbar_13/CMSSW_7_4_0_pre5-MCRUN2_73_V7-v1/AODSIM
filesRelValProdTTbarAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_4_0_pre5'
                        #, relVal        = 'RelValProdTTbar_13'
                        #, globalTag     = 'MCRUN2_73_V7'
                        #, dataTier      = 'AODSIM'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/MINIAODSIM/MCRUN2_74_V1-v1/00000/58D65E53-E5A8-E411-BC5E-002354EF3BDF.root'
    )

# /RelValProdTTbar_13/CMSSW_7_4_0_pre5-MCRUN2_73_V7-v1/GEN-SIM-RECO
filesRelValProdTTbarGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_4_0_pre5'
                        #, relVal        = 'RelValProdTTbar_13'
                        #, globalTag     = 'MCRUN2_73_V7'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-RECO/MCRUN2_74_V1-v1/00000/2E97B200-D0A8-E411-BE99-0025905A60B8.root'
    )

# /RelValTTbar/CMSSW_7_4_0_pre5-PU_MCRUN1_73_V2_FastSim-v1/GEN-SIM-DIGI-RECO
filesRelValTTbarPileUpFastSimGENSIMDIGIRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_4_0_pre5'
                        #, relVal        = 'RelValTTbar'
                        #, globalTag     = 'PU_MCRUN1_73_V2_FastSim'
                        #, dataTier      = 'GEN-SIM-DIGI-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13_AVE10/GEN-SIM-DIGI-RECO/PU25ns_MCRUN2_74_V1_FastSim-v1/00000/004A7A7A-AFA8-E411-9676-002618943913.root'
    )

# /RelValTTbar_13/CMSSW_7_4_0_pre5-PU25ns_MCRUN2_73_V7-v1/GEN-SIM-RECO
filesRelValTTbarPileUpGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_4_0_pre5'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU25ns_MCRUN2_73_V7'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_4_0_pre6/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V1-v3/00000/067739D0-AFAB-E411-AC03-0025905A48D0.root'
    )

# /SingleMu/CMSSW_7_4_0_pre5-GR_R_73_V0A_RelVal_mu2012D-v1/MINIAOD
filesRelValSingleMuMINIAOD = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_4_0_pre5'
                        #, relVal        = 'SingleMu'
                        #, globalTag     = 'GR_R_73_V0A_RelVal_mu2012D'
                        #, dataTier      = 'MINIAOD'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_4_0_pre6/SingleMu/MINIAOD/GR_R_74_V0A_RelVal_mu2012D-v1/00000/2AAE1C10-43A9-E411-B7FF-0025905A611C.root'
    )

# /SingleMu/CMSSW_7_4_0_pre5-GR_R_73_V0A_RelVal_mu2012D-v1/RECO
filesSingleMuRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_4_0_pre5'
                        #, relVal        = 'SingleMu'
                        #, dataTier      = 'RECO'
                        #, globalTag     = 'GR_R_73_V0A_RelVal_mu2012D'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_4_0_pre6/SingleMu/RECO/GR_R_74_V0A_RelVal_mu2012D-v1/00000/000D5DD4-F0A8-E411-95AC-0025905B85AE.root'
    )
