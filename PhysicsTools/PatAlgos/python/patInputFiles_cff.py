import FWCore.ParameterSet.Config as cms
from PhysicsTools.PatAlgos.tools.cmsswVersionTools import pickRelValInputFiles

filesRelValProdTTbarAODSIM = cms.untracked.vstring(
    pickRelValInputFiles( cmsswVersion  = 'CMSSW_6_1_1'
                        , relVal        = 'RelValProdTTbar'
                        , globalTag     = 'START61_V11'
                        , dataTier      = 'AODSIM'
                        , maxVersions   = 1
                        , numberOfFiles = 1
                        )
    )
filesRelValProdTTbarGENSIMRECO = cms.untracked.vstring(
    pickRelValInputFiles( cmsswVersion  = 'CMSSW_6_1_1'
                        , relVal        = 'RelValProdTTbar'
                        , globalTag     = 'START61_V11'
                        , dataTier      = 'GEN-SIM-RECO'
                        , maxVersions   = 1
                        , numberOfFiles = 1
                        )
    )
filesRelValTTbarPileUpGENSIMRECO = cms.untracked.vstring(
    pickRelValInputFiles( cmsswVersion  = 'CMSSW_6_1_1'
                        , relVal        = 'RelValTTbar'
                        , globalTag     = 'PU_START61_V11'
                        , dataTier      = 'GEN-SIM-RECO'
                        , maxVersions   = 1
                        , numberOfFiles = 1
                        )
    )
filesSingleMuRECO = cms.untracked.vstring(
    #pickRelValInputFiles( cmsswVersion  = 'CMSSW_6_1_1'
                        #, relVal        = 'SingleMu'
                        #, dataTier      = 'RECO'
                        #, globalTag     = 'GR_R_61_V6_RelVal_mu2012C'
                        #, maxVersions   = 1
                        #, numberOfFiles = 1
                        #)
    # only one block available at CERN
    # FIXME: need to fix DBS query in 'pickRelValInputFiles' to identify them properly
    # ==> query for file requiring dataset AND site does not work in DBS :-(
      '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/02A67740-E076-E211-9305-003048678FF8.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/1E082128-F476-E211-BB0A-0025905964B2.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/1E0935CA-F176-E211-A643-002618943908.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/2A9EE011-F476-E211-8953-002618943901.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/2C717083-0477-E211-AB6B-003048678B14.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/2CC1BE13-F476-E211-8B15-00261894387C.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/4432AED5-F076-E211-A5D8-002590596498.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/483ECF04-F076-E211-96B6-003048FFD730.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/505EB6C1-0077-E211-9FE8-003048FFD744.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/529E3F11-F076-E211-9868-003048D15E24.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/58185BB7-0377-E211-AFD3-00304867BFAA.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/5C7567CC-E076-E211-B680-00248C55CC62.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/681E2BEB-F876-E211-B231-002618943930.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/684AB1A6-0377-E211-8EF3-002618943964.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/6AFB301D-F176-E211-A7D4-0025905964BE.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/6C7339ED-F876-E211-B695-002618943956.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/725EE818-F176-E211-90F1-003048678B3C.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/74787FF8-EF76-E211-A57A-0026189438FC.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/781A642B-F176-E211-813E-003048FFCB9E.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/7C148207-F976-E211-869E-002618943972.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/7CBBBDD3-0377-E211-93B2-003048FFD7BE.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/920B11DF-F876-E211-A07D-0026189438AD.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/929956D1-F176-E211-8093-002618943977.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/96319AB0-0077-E211-8E6E-003048678FE0.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/9C7EDFC2-0077-E211-A619-003048FFD7C2.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/A0463FDD-F176-E211-92FB-003048FFCB84.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/A62B81DA-E076-E211-97FF-0025905964C0.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/A6F907CF-F176-E211-AEB3-003048678F62.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/AEA4B4C4-0077-E211-8432-003048FFD7C2.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/D01E6277-E676-E211-93BE-00261894397D.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/E4C17C96-F076-E211-987B-00261894383B.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/E8A22CD3-F176-E211-ADBD-00261894395B.root'
    , '/store/relval/CMSSW_6_1_1-GR_R_61_V6_RelVal_mu2012C/SingleMu/RECO/v1/00000/FA2B52BE-F076-E211-8BF0-0026189438AB.root'
    )
