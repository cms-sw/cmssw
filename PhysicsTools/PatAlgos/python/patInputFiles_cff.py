import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

# /RelValProdTTbar/CMSSW_7_1_0_pre4_AK4-START71_V1-v2/AODSIM
filesRelValProdTTbarAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_1_0_pre4_AK4'
                        #, relVal        = 'RelValProdTTbar'
                        #, globalTag     = 'START71_V1'
                        #, dataTier      = 'AODSIM'
                        #, maxVersions   = 2
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_1_0_pre4_AK4/RelValProdTTbar/AODSIM/START71_V1-v2/00000/7A3637AA-28B5-E311-BC25-003048678B94.root'
    )

# /RelValProdTTbar/CMSSW_7_1_0_pre4_AK4-START71_V1-v2/GEN-SIM-RECO
filesRelValProdTTbarGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_1_0_pre4_AK4'
                        #, relVal        = 'RelValProdTTbar'
                        #, globalTag     = 'START71_V1'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 2
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_1_0_pre4_AK4/RelValProdTTbar/GEN-SIM-RECO/START71_V1-v2/00000/04DF20AC-28B5-E311-814A-003048679296.root'
    )

# /RelValTTbar/CMSSW_7_1_0_pre4_AK4-PU_START71_V1_FastSim-v2/GEN-SIM-DIGI-RECO
filesRelValTTbarPileUpFastSimGENSIMDIGIRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_1_0_pre4_AK4'
                        #, relVal        = 'RelValTTbar'
                        #, globalTag     = 'PU_START71_V1_FastSim'
                        #, dataTier      = 'GEN-SIM-DIGI-RECO'
                        #, maxVersions   = 2
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_1_0_pre4_AK4/RelValTTbar/GEN-SIM-DIGI-RECO/PU_START71_V1_FastSim-v2/00000/04149ADF-FCB3-E311-8707-0025904C6788.root'
    )

# /RelValTTbar_13/CMSSW_7_1_0_pre4_AK4-PU50ns_POSTLS171_V2-v2/GEN-SIM-RECO
filesRelValTTbarPileUpGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_1_0_pre4_AK4'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU50ns_POSTLS171_V2'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 2
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_1_0_pre4_AK4/RelValTTbar_13/GEN-SIM-RECO/PU50ns_POSTLS171_V2-v2/00000/1487CD0E-A4B3-E311-96D2-0025904C678A.root'
    )

# /SingleMu/CMSSW_7_1_0_pre4_AK4-GR_R_71_V1_RelVal_mu2012D-v2/RECO
filesSingleMuRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_1_0_pre4_AK4' # not at CERN
                        #, relVal        = 'SingleMu'
                        #, dataTier      = 'RECO'
                        #, globalTag     = 'GR_R_71_V1_RelVal_mu2012D'
                        #, maxVersions   = 2
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_1_0_pre4_AK4/SingleMu/RECO/GR_R_71_V1_RelVal_mu2012D-v2/00000/00255F2D-36B5-E311-9749-0025905AA9CC.root'
    )
