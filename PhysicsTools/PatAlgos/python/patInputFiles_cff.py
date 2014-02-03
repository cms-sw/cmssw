import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

filesRelValProdTTbarAODSIM = cms.untracked.vstring(
    pickRelValInputFiles( cmsswVersion  = 'CMSSW_6_2_0'
                        , relVal        = 'RelValProdTTbar'
                        , globalTag     = 'PRE_ST62_V8'
                        , dataTier      = 'AODSIM'
                        , maxVersions   = 3
                        , numberOfFiles = 1
                        )
    )
filesRelValProdTTbarGENSIMRECO = cms.untracked.vstring(
    pickRelValInputFiles( cmsswVersion  = 'CMSSW_6_2_0'
                        , relVal        = 'RelValProdTTbar'
                        , globalTag     = 'PRE_ST62_V8'
                        , dataTier      = 'GEN-SIM-RECO'
                        , maxVersions   = 3
                        , numberOfFiles = 1
                        )
    )
filesRelValTTbarPileUpGENSIMRECO = cms.untracked.vstring(
    pickRelValInputFiles( cmsswVersion  = 'CMSSW_6_2_0'
                        , relVal        = 'RelValTTbar'
                        , globalTag     = 'PU_PRE_ST62_V8'
                        , dataTier      = 'GEN-SIM-RECO'
                        , maxVersions   = 2
                        , numberOfFiles = 1
                        )
    )
filesSingleMuRECO = cms.untracked.vstring(
    pickRelValInputFiles( cmsswVersion  = 'CMSSW_6_2_0_pre8'
                        , relVal        = 'SingleMu'
                        , dataTier      = 'RECO'
                        , globalTag     = 'PRE_62_V8_RelVal_mu2012D'
                        , maxVersions   = 1
                        , numberOfFiles = 1
                        )
    # only one block available at CERN
    # FIXME: need to fix DBS query in 'pickRelValInputFiles' to identify them properly
    # ==> query for file requiring dataset AND site does not work in DBS :-(
       #'/store/relval/CMSSW_6_2_0_pre8/SingleMu/RECO/PRE_62_V8_RelVal_mu2012B-v1/00000/029F8FA5-D7E0-E211-BCCF-001E67398430.root',
       #'/store/relval/CMSSW_6_2_0_pre8/SingleMu/RECO/PRE_62_V8_RelVal_mu2012B-v1/00000/1A9A0FE7-D5E0-E211-9868-003048F01164.root',
       #'/store/relval/CMSSW_6_2_0_pre8/SingleMu/RECO/PRE_62_V8_RelVal_mu2012B-v1/00000/4E0F44F0-D5E0-E211-9A0A-003048CF6780.root',
       #'/store/relval/CMSSW_6_2_0_pre8/SingleMu/RECO/PRE_62_V8_RelVal_mu2012B-v1/00000/6E43C4C0-DBE0-E211-B452-003048D37366.root',
       #'/store/relval/CMSSW_6_2_0_pre8/SingleMu/RECO/PRE_62_V8_RelVal_mu2012B-v1/00000/A426EC8A-DAE0-E211-BE58-D8D385FF4A94.root',
       #'/store/relval/CMSSW_6_2_0_pre8/SingleMu/RECO/PRE_62_V8_RelVal_mu2012B-v1/00000/A45F6ADF-D8E0-E211-948B-003048FE9D54.root',
       #'/store/relval/CMSSW_6_2_0_pre8/SingleMu/RECO/PRE_62_V8_RelVal_mu2012B-v1/00000/CE1D92A1-D9E0-E211-B7F0-C860001BD934.root',
       #'/store/relval/CMSSW_6_2_0_pre8/SingleMu/RECO/PRE_62_V8_RelVal_mu2012B-v1/00000/D61C12DD-E9E0-E211-8A9E-5404A63886D2.root',
       #'/store/relval/CMSSW_6_2_0_pre8/SingleMu/RECO/PRE_62_V8_RelVal_mu2012B-v1/00000/E4F44285-DFE0-E211-BEA9-0025B3203918.root',
       #'/store/relval/CMSSW_6_2_0_pre8/SingleMu/RECO/PRE_62_V8_RelVal_mu2012B-v1/00000/F848CDC3-D6E0-E211-BA88-003048F009C4.root'
    )
