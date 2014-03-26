import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

filesRelValProdTTbarAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_6_2_0_pre5' # event content in 620pre8 broken
                        #, relVal        = 'RelValProdTTbar'
                        #, globalTag     = 'PRE_ST61_V1'
                        #, dataTier      = 'AODSIM'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_6_2_0_pre5/RelValProdTTbar/AODSIM/PRE_ST61_V1-v1/00000/3C9BFDAA-79A4-E211-BC9F-C86000151B96.root'
    )
filesRelValProdTTbarGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_6_2_0_pre5' # event content in 620pre8 broken
                        #, relVal        = 'RelValProdTTbar'
                        #, globalTag     = 'PRE_ST61_V1'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_6_2_0_pre5/RelValProdTTbar/GEN-SIM-RECO/PRE_ST61_V1-v1/00000/EA73A371-77A4-E211-98E0-003048F0E320.root'
    )
filesRelValTTbarPileUpGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_6_2_0_pre5' # event content in 620pre8 broken
                        #, relVal        = 'RelValTTbar'
                        #, globalTag     = 'PU_PRE_ST61_V1'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_6_2_0_pre5/RelValTTbar/GEN-SIM-RECO/PU_PRE_ST61_V1-v1/00000/E877A0D2-DEA5-E211-9EA4-002481E0D73C.root'
    )
filesSingleMuRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_6_2_0_pre8'
                        #, relVal        = 'SingleMu'
                        #, dataTier      = 'RECO'
                        #, globalTag     = 'PRE_62_V8_RelVal_mu2012D'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_6_2_0_pre8/SingleMu/RECO/PRE_62_V8_RelVal_mu2012D-v1/00001/FCF80FBD-ECDF-E211-A99E-001E6739721F.root'
    )
