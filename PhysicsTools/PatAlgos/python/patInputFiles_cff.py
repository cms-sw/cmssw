import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

# /RelValProdTTbar/CMSSW_7_1_0_pre9-START71_V5-v1/AODSIM
filesRelValProdTTbarAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_1_0_pre9'
                        #, relVal        = 'RelValProdTTbar'
                        #, globalTag     = 'START71_V5'
                        #, dataTier      = 'AODSIM'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_1_0_pre9/RelValProdTTbar/AODSIM/START71_V5-v1/00000/F4F6CBF6-8AF0-E311-8AEF-003048678ED2.root'
    )

# /RelValProdTTbar/CMSSW_7_1_0_pre9-START71_V5-v1/GEN-SIM-RECO
filesRelValProdTTbarGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_1_0_pre9'
                        #, relVal        = 'RelValProdTTbar'
                        #, globalTag     = 'START71_V5'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_1_0_pre9/RelValProdTTbar/GEN-SIM-RECO/START71_V5-v1/00000/2622F834-78F0-E311-A9BE-003048678B7C.root'
    )

# /RelValTTbar/CMSSW_7_1_0-PU_START71_V7_FastSim-v1/GEN-SIM-DIGI-RECO
filesRelValTTbarPileUpFastSimGENSIMDIGIRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_1_0_pre9'
                        #, relVal        = 'RelValTTbar'
                        #, globalTag     = 'PU_START71_V5_FastSim'
                        #, dataTier      = 'GEN-SIM-DIGI-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_1_0_pre9/RelValTTbar/GEN-SIM-DIGI-RECO/PU_START71_V5_FastSim-v1/00000/0022BD0D-BAF0-E311-922A-02163E00F4EB.root'
    )

# /RelValTTbar_13/CMSSW_7_1_0-PU50ns_POSTLS171_V16-v1/GEN-SIM-RECO
filesRelValTTbarPileUpGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_1_0'
                        #, relVal        = 'RelValTTbar_13'
                        #, globalTag     = 'PU50ns_POSTLS171_V16'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_1_0/RelValTTbar_13/GEN-SIM-RECO/PU50ns_POSTLS171_V16-v1/00000/388EB19E-FAFE-E311-A39D-0025905A60EE.root'
    )

# /SingleMu/CMSSW_7_1_0_pre9-GR_R_71_V4_RelVal_mu2012D-v1/RECO
filesSingleMuRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_7_1_0_pre9' # not at CERN
                        #, relVal        = 'SingleMu'
                        #, dataTier      = 'RECO'
                        #, globalTag     = 'GR_R_71_V4_RelVal_mu2012D'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_7_1_0_pre9/SingleMu/RECO/GR_R_71_V4_RelVal_mu2012D-v1/00000/0055BBC8-C3F0-E311-BBBE-0025905A605E.root'
    )
