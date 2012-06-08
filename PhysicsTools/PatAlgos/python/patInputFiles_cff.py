import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

filesRelValProdTTbarAODSIM = cms.untracked.vstring(
#     pickRelValInputFiles( cmsswVersion  = 'CMSSW_5_2_0'
#                         , relVal        = 'RelValProdTTbar'
#                         , globalTag     = 'START52_V4A'
#                         , dataTier      = 'AODSIM'
#                         , maxVersions   = 1
#                         , numberOfFiles = 1
#                         )
    '/store/relval/CMSSW_5_2_0/RelValProdTTbar/AODSIM/START52_V4A-v1/0250/68FCD498-F969-E111-9366-002618943949.root'
    )
filesRelValProdTTbarGENSIMRECO = cms.untracked.vstring(
#     pickRelValInputFiles( cmsswVersion  = 'CMSSW_5_2_0'
#                         , relVal        = 'RelValProdTTbar'
#                         , globalTag     = 'START52_V4A'
#                         , dataTier      = 'GEN-SIM-RECO'
#                         , maxVersions   = 1
#                         , numberOfFiles = 1
#                         )
    '/store/relval/CMSSW_5_2_0/RelValProdTTbar/GEN-SIM-RECO/START52_V4A-v1/0247/0AA219D0-FA68-E111-AD2C-001A92971BA0.root',
    '/store/relval/CMSSW_5_2_0/RelValProdTTbar/GEN-SIM-RECO/START52_V4A-v1/0247/440ECFBB-F968-E111-BE24-002618943974.root',
    '/store/relval/CMSSW_5_2_0/RelValProdTTbar/GEN-SIM-RECO/START52_V4A-v1/0248/5CCB0F34-E869-E111-906C-002618943865.root',
    '/store/relval/CMSSW_5_2_0/RelValProdTTbar/GEN-SIM-RECO/START52_V4A-v1/0248/7ABCC6E5-FD68-E111-9C8E-002618943920.root'
    )
filesSingleMuRECO = cms.untracked.vstring(
#     pickRelValInputFiles( cmsswVersion  = 'CMSSW_5_2_0'
#                         , relVal        = 'SingleMu'
#                         , dataTier      = 'RECO'
#                         , globalTag     = 'GR_R_52_V4_RelVal_mu2011B'
#                         , maxVersions   = 1
#                         , numberOfFiles = 1
#                         )
    '/store/relval/CMSSW_5_2_0/SingleMu/RECO/GR_R_52_V4_RelVal_mu2011B-v1/0000/A61F10B1-A469-E111-A19A-003048678B14.root',
    '/store/relval/CMSSW_5_2_0/SingleMu/RECO/GR_R_52_V4_RelVal_mu2011B-v1/0000/B0380C1D-A169-E111-B187-0026189438E1.root',
    '/store/relval/CMSSW_5_2_0/SingleMu/RECO/GR_R_52_V4_RelVal_mu2011B-v1/0000/CEC5AF39-A969-E111-9348-003048679162.root',
    '/store/relval/CMSSW_5_2_0/SingleMu/RECO/GR_R_52_V4_RelVal_mu2011B-v1/0000/D2FD6163-E268-E111-B454-003048FFD752.root',
    '/store/relval/CMSSW_5_2_0/SingleMu/RECO/GR_R_52_V4_RelVal_mu2011B-v1/0247/E62D0B87-F468-E111-8E55-003048FFCB6A.root',
    '/store/relval/CMSSW_5_2_0/SingleMu/RECO/GR_R_52_V4_RelVal_mu2011B-v1/0000/1A5AE45F-E268-E111-BCD6-0026189438EA.root',
    '/store/relval/CMSSW_5_2_0/SingleMu/RECO/GR_R_52_V4_RelVal_mu2011B-v1/0000/3CE2DD96-A269-E111-9B29-002618943908.root',
    '/store/relval/CMSSW_5_2_0/SingleMu/RECO/GR_R_52_V4_RelVal_mu2011B-v1/0000/6620735F-E268-E111-BD0C-0026189438CB.root',
    '/store/relval/CMSSW_5_2_0/SingleMu/RECO/GR_R_52_V4_RelVal_mu2011B-v1/0000/763A095E-E268-E111-B91C-002618943946.root',
    '/store/relval/CMSSW_5_2_0/SingleMu/RECO/GR_R_52_V4_RelVal_mu2011B-v1/0000/9AB2DDCD-AB69-E111-9F33-002618943910.root',
    '/store/relval/CMSSW_5_2_0/SingleMu/RECO/GR_R_52_V4_RelVal_mu2011B-v1/0000/00C45F5D-E268-E111-925D-00248C0BE014.root',
    '/store/relval/CMSSW_5_2_0/SingleMu/RECO/GR_R_52_V4_RelVal_mu2011B-v1/0000/068EAB5F-E268-E111-B733-002618FDA237.root',
    '/store/relval/CMSSW_5_2_0/SingleMu/RECO/GR_R_52_V4_RelVal_mu2011B-v1/0000/0824DE58-E268-E111-B35E-0026189438FE.root'
    )
