import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

filesRelValProdTTbarAODSIM = cms.untracked.vstring(
    pickRelValInputFiles( cmsswVersion  = 'CMSSW_5_2_0_pre6'
                        , relVal        = 'RelValProdTTbar'
                        , globalTag     = 'START52_V2'
                        , dataTier      = 'AODSIM'
                        , maxVersions   = 3
                        , numberOfFiles = 1
                        )
    )
filesRelValProdTTbarGENSIMRECO = cms.untracked.vstring(
    pickRelValInputFiles( cmsswVersion  = 'CMSSW_5_2_0_pre6'
                        , relVal        = 'RelValProdTTbar'
                        , globalTag     = 'START52_V2'
                        , dataTier      = 'GEN-SIM-RECO'
                        , maxVersions   = 3
                        , numberOfFiles = 1
                        )
    )
filesSingleMuRECO = cms.untracked.vstring(
    pickRelValInputFiles( cmsswVersion  = 'CMSSW_5_2_0_pre6'
                        , relVal        = 'SingleMu'
                        , dataTier      = 'RECO'
                        , globalTag     = 'GR_R_52_V3_RelVal_mu2011B'
                        , maxVersions   = 3
                        , numberOfFiles = 1
                        )
    )
