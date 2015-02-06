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
        '/store/relval/CMSSW_7_4_0_pre5/RelValTTbar_13/MINIAODSIM/PU25ns_MCRUN2_73_V7-v1/00000/B626832E-D6A0-E411-B0D9-0025905938A8.root'
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
        '/store/relval/CMSSW_7_4_0_pre5/RelValProdTTbar_13/AODSIM/MCRUN2_73_V7-v1/00000/621B749D-F19D-E411-896C-003048FFCBA4.root'
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
        '/store/relval/CMSSW_7_4_0_pre5/RelValProdTTbar_13/GEN-SIM-RECO/MCRUN2_73_V7-v1/00000/4EBA2C0B-EB9D-E411-810A-002618FDA248.root'
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
        '/store/relval/CMSSW_7_4_0_pre5/RelValTTbar/GEN-SIM-DIGI-RECO/PU_MCRUN1_73_V2_FastSim-v1/00000/00024C55-D39D-E411-969A-0025905A60C6.root'
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
        '/store/relval/CMSSW_7_4_0_pre5/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_73_V7-v1/00000/0A054268-C9A0-E411-8CE6-0025905A6134.root'
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
        '/store/relval/CMSSW_7_4_0_pre5/SingleMu/MINIAOD/GR_R_73_V0A_RelVal_mu2012D-v1/00000/0855341F-5C9E-E411-9A51-0025905A60CA.root'
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
        '/store/relval/CMSSW_7_4_0_pre5/SingleMu/RECO/GR_R_73_V0A_RelVal_mu2012D-v1/00000/00005DA8-EE9D-E411-A0B2-002618FDA277.root'
    )
