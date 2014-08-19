import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

# /RelValProdTTbar/CMSSW_7_2_0_pre1-START72_V1-v1/AODSIM
filesRelValProdTTbarAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_2_0_pre1'
                        #, relVal        = 'RelValProdTTbar'
                        #, globalTag     = 'START72_V1'
                        #, dataTier      = 'AODSIM'
                        #, maxVersions   = 2
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_2_0_pre1/RelValProdTTbar/AODSIM/START72_V1-v1/00000/7469A167-12FE-E311-BE14-003048FFD728.root'
    )

# /RelValProdTTbar/CMSSW_7_2_0_pre1-START72_V1-v1/GEN-SIM-RECO
filesRelValProdTTbarGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_2_0_pre1'
                        #, relVal        = 'RelValProdTTbar'
                        #, globalTag     = 'START72_V1'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 2
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_2_0_pre1/RelValProdTTbar/GEN-SIM-RECO/START72_V1-v1/00000/02F99F3C-03FE-E311-B8D5-0025905AA9CC.root'
    )

# /RelValTTbar/CMSSW_7_2_0_pre1-PU_START72_V1_FastSim-v1/GEN-SIM-DIGI-RECO
filesRelValTTbarPileUpFastSimGENSIMDIGIRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_2_0_pre1'
                        #, relVal        = 'RelValTTbar'
                        #, globalTag     = 'PU_START72_V1_FastSim'
                        #, dataTier      = 'GEN-SIM-DIGI-RECO'
                        #, maxVersions   = 2
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_2_0_pre1/RelValTTbar/GEN-SIM-DIGI-RECO/PU_START72_V1_FastSim-v1/00000/002EC191-6EFD-E311-8707-0025905B85D0.root'
    )

# /RelValTTbar_13/CMSSW_7_2_0_pre1-PU50ns_POSTLS172_V2-v1/GEN-SIM-RECO
filesRelValTTbarPileUpGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_2_0_pre1'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU50ns_POSTLS172_V2'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 2
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_2_0_pre1/RelValTTbar_13/GEN-SIM-RECO/PU50ns_POSTLS172_V2-v1/00000/0AA51FF6-8EFD-E311-B591-0025905A6068.root'
    )

# /SingleMu/CMSSW_7_2_0_pre1-GR_R_72_V1_RelVal_mu2012D-v1/RECO
filesSingleMuRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_2_0_pre1'
                        #, relVal        = 'SingleMu'
                        #, dataTier      = 'RECO'
                        #, globalTag     = 'GR_R_72_V1_RelVal_mu2012D'
                        #, maxVersions   = 2
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_2_0_pre1/SingleMu/RECO/GR_R_72_V1_RelVal_mu2012D-v1/00000/002D1019-ADFD-E311-9230-00259059642A.root'
    )
