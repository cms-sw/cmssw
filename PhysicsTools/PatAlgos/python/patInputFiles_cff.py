import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

filesRelValProdTTbarAODSIM = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_5_3_6'
                        #, relVal        = 'RelValProdTTbar'
                        #, globalTag     = 'START53_V14'
                        #, dataTier      = 'AODSIM'
                        #, maxVersions   = 2
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_5_3_6-START53_V14/RelValProdTTbar/AODSIM/v2/00000/76ED0FA6-1E2A-E211-B8F1-001A92971B72.root'
    )
filesRelValProdTTbarGENSIMRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_5_3_6'
                        #, relVal        = 'RelValProdTTbar'
                        #, globalTag     = 'START53_V14'
                        #, dataTier      = 'GEN-SIM-RECO'
                        #, maxVersions   = 2
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_5_3_6-START53_V14/RelValProdTTbar/GEN-SIM-RECO/v2/00000/B86B2DE8-122A-E211-AD41-003048678B84.root'
    )
filesSingleMuRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_5_3_6'
                        #, relVal        = 'SingleMu'
                        #, dataTier      = 'RECO'
                        #, globalTag     = 'GR_R_53_V15_RelVal_mu2012B'
                        #, maxVersions   = 2
                        #, numberOfFiles = 1
                        #, useDAS        = True
                        #)
        '/store/relval/CMSSW_5_3_6-GR_R_53_V15_RelVal_mu2012B/SingleMu/RECO/v2/00000/FADB72FB-EE29-E211-8678-003048678ADA.root'
    )
