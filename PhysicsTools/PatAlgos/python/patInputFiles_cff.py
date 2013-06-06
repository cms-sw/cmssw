import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

filesRelValProdTTbarAODSIM = cms.untracked.vstring(
    pickRelValInputFiles( cmsswVersion  = 'CMSSW_5_3_6'
                        , relVal        = 'RelValProdTTbar'
                        , globalTag     = 'START53_V14'
                        , dataTier      = 'AODSIM'
                        , maxVersions   = 2
                        , numberOfFiles = 1
                        )
    )
filesRelValProdTTbarGENSIMRECO = cms.untracked.vstring(
    pickRelValInputFiles( cmsswVersion  = 'CMSSW_5_3_6'
                        , relVal        = 'RelValProdTTbar'
                        , globalTag     = 'START53_V14'
                        , dataTier      = 'GEN-SIM-RECO'
                        , maxVersions   = 2
                        , numberOfFiles = 1
                        )
    )
filesSingleMuRECO = cms.untracked.vstring(
    pickRelValInputFiles( cmsswVersion  = 'CMSSW_5_3_6'
                        , relVal        = 'SingleMu'
                        , dataTier      = 'RECO'
                        , globalTag     = 'GR_R_53_V15_RelVal_mu2012B'
                        , maxVersions   = 2
                        , numberOfFiles = 1
                        )
    )
