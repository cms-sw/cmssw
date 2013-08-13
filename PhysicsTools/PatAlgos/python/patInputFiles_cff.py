import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

filesRelValProdTTbarAODSIM = cms.untracked.vstring(
    pickRelValInputFiles( cmsswVersion  = 'CMSSW_6_2_0_pre5'
                        , relVal        = 'RelValProdTTbar'
                        , globalTag     = 'PRE_ST61_V1'
                        , dataTier      = 'AODSIM'
                        , maxVersions   = 1
                        , numberOfFiles = 1
                        )
    )
filesRelValProdTTbarGENSIMRECO = cms.untracked.vstring(
    pickRelValInputFiles( cmsswVersion  = 'CMSSW_6_2_0_pre5'
                        , relVal        = 'RelValProdTTbar'
                        , globalTag     = 'PRE_ST61_V1'
                        , dataTier      = 'GEN-SIM-RECO'
                        , maxVersions   = 1
                        , numberOfFiles = 1
                        )
    )
filesRelValTTbarPileUpGENSIMRECO = cms.untracked.vstring(
    pickRelValInputFiles( cmsswVersion  = 'CMSSW_6_2_0_pre5'
                        , relVal        = 'RelValTTbar'
                        , globalTag     = 'PU_PRE_ST61_V1'
                        , dataTier      = 'GEN-SIM-RECO'
                        , maxVersions   = 1
                        , numberOfFiles = 1
                        )
    )
filesSingleMuRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_6_2_0_pre4'
                        #, relVal        = 'SingleMu'
                        #, dataTier      = 'RECO'
                        #, globalTag     = 'PRE_61_V1_RelVal_mu2012D'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #)
    # only one block available at CERN
    # FIXME: need to fix DBS query in 'pickRelValInputFiles' to identify them properly
    # ==> query for file requiring dataset AND site does not work in DBS :-(
       '/store/relval/CMSSW_6_2_0_pre4/SingleMu/RECO/PRE_61_V1_RelVal_mu2012D-v1/00000/2A8716C9-3492-E211-8E04-0025905938A4.root',
    )
