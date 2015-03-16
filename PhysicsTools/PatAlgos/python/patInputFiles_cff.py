import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

# /RelValTTbar_13/CMSSW_7_4_0_pre8-PU25ns_MCRUN2_74_V7-v1/MINIAODSIM
filesRelValTTbarPileUpMINIAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_4_0_pre8'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU25ns_MCRUN2_74_V7'
                        #, dataTier      = 'MINIAODSIM'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_4_0_pre8/RelValTTbar_13/MINIAODSIM/PU25ns_MCRUN2_74_V7-v1/00000/122AA6C7-6BBD-E411-80C1-002590593902.root'
    )

# /RelValProdTTbar_13/CMSSW_7_4_0_pre8-MCRUN2_74_V7-v1/AODSIM
filesRelValProdTTbarAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_4_0_pre8'
                        #, relVal        = 'RelValProdTTbar_13'
                        #, globalTag     = 'MCRUN2_74_V7'
                        #, dataTier      = 'AODSIM'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_4_0_pre8/RelValProdTTbar_13/AODSIM/MCRUN2_74_V7-v1/00000/44E1E4BA-50BD-E411-A57A-002618943949.root'
    )

# /RelValProdTTbar_13/CMSSW_7_4_0_pre8-MCRUN2_74_V7-v1/GEN-SIM-RECO
filesRelValProdTTbarGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_4_0_pre8'
                        #, relVal        = 'RelValProdTTbar_13'
                        #, globalTag     = 'MCRUN2_74_V7'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_4_0_pre8/RelValProdTTbar_13/GEN-SIM-RECO/MCRUN2_74_V7-v1/00000/32B61181-4FBD-E411-A930-0026189438F6.root'
    )

# /RelValTTbar_13/CMSSW_7_4_0_pre8-MCRUN2_74_V7_FastSim-v1/GEN-SIM-DIGI-RECO
filesRelValTTbarFastSimGENSIMDIGIRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_4_0_pre8'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'MCRUN2_74_V7_FastSim'
                        #, dataTier      = 'GEN-SIM-DIGI-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_4_0_pre8/RelValTTbar_13/GEN-SIM-DIGI-RECO/MCRUN2_74_V7_FastSim-v1/00000/008BD645-1EBD-E411-B3D0-003048FFD740.root'
    )

# /RelValTTbar_13/CMSSW_7_4_0_pre8-PU25ns_MCRUN2_74_V7-v1/GEN-SIM-RECO
filesRelValTTbarPileUpGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_4_0_pre8'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU25ns_MCRUN2_74_V7'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_4_0_pre8/RelValTTbar_13/GEN-SIM-RECO/PU25ns_MCRUN2_74_V7-v1/00000/10D198BE-64BD-E411-A4E4-00248C55CC40.root'
    )

# /SingleMu/CMSSW_7_4_0_pre8-GR_R_74_V8A_RelVal_mu2012D-v1/MINIAOD
filesRelValSingleMuMINIAOD = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_4_0_pre8'
                        #, relVal        = 'SingleMu'
                        #, globalTag     = 'GR_R_74_V8A_RelVal_mu2012D'
                        #, dataTier      = 'MINIAOD'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_4_0_pre8/SingleMu/MINIAOD/GR_R_74_V8A_RelVal_mu2012D-v1/00000/2494813A-ECBD-E411-ADDB-0025905A48FC.root'
    )

# /SingleMu/CMSSW_7_4_0_pre8-GR_R_74_V8A_RelVal_mu2012D-v1/RECO
filesSingleMuRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_4_0_pre8'
                        #, relVal        = 'SingleMu'
                        #, dataTier      = 'RECO'
                        #, globalTag     = 'GR_R_74_V8A_RelVal_mu2012D'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_4_0_pre8/SingleMu/RECO/GR_R_74_V8A_RelVal_mu2012D-v1/00000/001F3051-89BD-E411-9DAA-0025905822B6.root'
    )
