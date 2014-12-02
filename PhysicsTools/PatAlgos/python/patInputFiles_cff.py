import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

# /RelValTTbar_13/CMSSW_7_3_0_pre3-PU50ns_MCRUN2_73_V4-v1/MINIAODSIM
filesRelValProdTTbarPileUpMINIAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_3_0_pre3'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU50ns_MCRUN2_73_V4'
                        #, dataTier      = 'MINIAODSIM'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_3_0_pre3/RelValTTbar_13/MINIAODSIM/PU50ns_MCRUN2_73_V4-v1/00000/8202F53B-E876-E411-A45A-02163E010F0C.root'
    )

# /RelValProdTTbar_13/CMSSW_7_3_0_pre3-MCRUN2_73_V5-v1/AODSIM
filesRelValProdTTbarAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_3_0_pre3'
                        #, relVal        = 'RelValProdTTbar_13'
                        #, globalTag     = 'MCRUN2_73_V5'
                        #, dataTier      = 'AODSIM'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_3_0_pre3/RelValProdTTbar_13/AODSIM/MCRUN2_73_V5-v1/00000/24901D70-7A76-E411-B2CE-02163E00FBB8.root'
    )

# /RelValProdTTbar_13/CMSSW_7_3_0_pre3-MCRUN2_73_V5-v1/GEN-SIM-RECO
filesRelValProdTTbarGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_3_0_pre3'
                        #, relVal        = 'RelValProdTTbar_13'
                        #, globalTag     = 'MCRUN2_73_V5'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_3_0_pre3/RelValProdTTbar_13/GEN-SIM-RECO/MCRUN2_73_V5-v1/00000/6C790212-5976-E411-8B03-02163E00E617.root'
    )

# /RelValTTbar/CMSSW_7_3_0_pre3-PU_MCRUN1_73_V1_FastSim-v1/GEN-SIM-DIGI-RECO
filesRelValTTbarPileUpFastSimGENSIMDIGIRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_3_0_pre3'
                        #, relVal        = 'RelValTTbar'
                        #, globalTag     = 'PU_MCRUN1_73_V1_FastSim'
                        #, dataTier      = 'GEN-SIM-DIGI-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_3_0_pre3/RelValTTbar/GEN-SIM-DIGI-RECO/PU_MCRUN1_73_V1_FastSim-v1/00000/00062E1F-1976-E411-A031-02163E00F50C.root'
    )

# /RelValTTbar_13/CMSSW_7_3_0_pre3-PU50ns_MCRUN2_73_V4-v1/GEN-SIM-RECO
filesRelValTTbarPileUpGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_3_0_pre3'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU50ns_MCRUN2_73_V4'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_3_0_pre3/RelValTTbar_13/GEN-SIM-RECO/PU50ns_MCRUN2_73_V4-v1/00000/0A59D5ED-7276-E411-9BAD-02163E010EC2.root'
    )

# /SingleMu/CMSSW_7_3_0_pre3-GR_R_73_V0A_RelVal_mu2012D-v1/MINIAOD
filesRelValSingleMuMINIAOD = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_3_0_pre3'
                        #, relVal        = 'SingleMu'
                        #, globalTag     = 'GR_R_73_V0A_RelVal_mu2012D'
                        #, dataTier      = 'MINIAOD'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_3_0_pre3/SingleMu/MINIAOD/GR_R_73_V0A_RelVal_mu2012D-v1/00000/126DA110-DB76-E411-AD6F-02163E010DC1.root'
    )

# /SingleMu/CMSSW_7_3_0_pre3-GR_R_73_V0A_RelVal_mu2012D-v1/RECO # not at CERN
filesSingleMuRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_3_0_pre3'
                        #, relVal        = 'SingleMu'
                        #, dataTier      = 'RECO'
                        #, globalTag     = 'GR_R_73_V0A_RelVal_mu2012D'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_3_0_pre3/SingleMu/RECO/GR_R_73_V0A_RelVal_mu2012D-v1/00000/00544682-1376-E411-8004-02163E00E66B.root'
    )
