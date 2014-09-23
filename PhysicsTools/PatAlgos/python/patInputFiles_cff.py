import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

# /RelValProdTTbar/CMSSW_7_2_0_pre5-START72_V1-v1/AODSIM
filesRelValProdTTbarAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_2_0_pre5'
                        #, relVal        = 'RelValProdTTbar'
                        #, globalTag     = 'START72_V1'
                        #, dataTier      = 'AODSIM'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_2_0_pre5/RelValProdTTbar/AODSIM/START72_V1-v1/00000/84686BF3-AC30-E411-B9A8-00261894391C.root'
    )

# /RelValProdTTbar/CMSSW_7_2_0_pre5-START72_V1-v1/GEN-SIM-RECO
filesRelValProdTTbarGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_2_0_pre5'
                        #, relVal        = 'RelValProdTTbar'
                        #, globalTag     = 'START72_V1'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_2_0_pre5/RelValProdTTbar/GEN-SIM-RECO/START72_V1-v1/00000/022350A9-AC30-E411-B225-0025905A6076.root'
    )

# /RelValTTbar/CMSSW_7_2_0_pre5-PU_START72_V1_FastSim-v1/GEN-SIM-DIGI-RECO
filesRelValTTbarPileUpFastSimGENSIMDIGIRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_2_0_pre5'
                        #, relVal        = 'RelValTTbar'
                        #, globalTag     = 'PU_START71_V5_FastSim'
                        #, dataTier      = 'GEN-SIM-DIGI-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_2_0_pre5/RelValTTbar/GEN-SIM-DIGI-RECO/PU_START72_V1_FastSim-v1/00000/0082D343-4B30-E411-93C1-0026189438F6.root'
    )

# /RelValTTbar_13/CMSSW_7_2_0_pre5-PU50ns_POSTLS172_V4-v1/GEN-SIM-RECO
filesRelValTTbarPileUpGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_2_0_pre5'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU50ns_POSTLS172_V4'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_2_0_pre5/RelValTTbar_13/GEN-SIM-RECO/PU50ns_POSTLS172_V4-v1/00000/303570F4-6030-E411-B7A6-0025905A60A0.root'
    )

# /SingleMu/CMSSW_7_2_0_pre5-GR_R_72_V2_RelVal_mu2012D-v1/RECO
filesSingleMuRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_2_0_pre5' # not at CERN
                        #, relVal        = 'SingleMu'
                        #, dataTier      = 'RECO'
                        #, globalTag     = 'GR_R_72_V2_RelVal_mu2012D'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_2_0_pre5/SingleMu/RECO/GR_R_72_V2_RelVal_mu2012D-v1/00000/002CC908-9130-E411-B785-003048FFD732.root'
    )
