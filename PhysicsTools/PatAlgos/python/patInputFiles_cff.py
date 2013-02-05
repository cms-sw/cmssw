import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

filesRelValProdTTbarAODSIM = cms.untracked.vstring(
    pickRelValInputFiles( cmsswVersion  = 'CMSSW_5_2_7'
                        , relVal        = 'RelValProdTTbar'
                        , globalTag     = 'START52_V10'
                        , dataTier      = 'AODSIM'
                        , maxVersions   = 1
                        , numberOfFiles = 1
                        )
    )
filesRelValProdTTbarGENSIMRECO = cms.untracked.vstring(
    pickRelValInputFiles( cmsswVersion  = 'CMSSW_5_2_7'
                        , relVal        = 'RelValProdTTbar'
                        , globalTag     = 'START52_V10'
                        , dataTier      = 'GEN-SIM-RECO'
                        , maxVersions   = 1
                        , numberOfFiles = 1
                        )
    )
filesSingleMuRECO = cms.untracked.vstring(
    pickRelValInputFiles( cmsswVersion  = 'CMSSW_5_2_7'
                        , relVal        = 'SingleMu'
                        , dataTier      = 'RECO'
                        , globalTag     = 'GR_R_52_V7_RelVal_mu2011B'
                        , maxVersions   = 1
                        , numberOfFiles = 1
                        )
    )
