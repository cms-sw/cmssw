# The following comments couldn't be translated into the new config version:

#! /bin/env cmsRun

#include "CondCore/DBCommon/data/CondDBCommon.cfi"
#replace CondDBCommon.connect = "sqlite_file:btagnew.db"
#replace CondDBCommon.catalog = "file:mycatalog.xml"
#        es_source = PoolDBESSource {
#                                  using CondDBCommon
#                                 VPSet toGet = {
#                                   {string record = "BTagTrackProbability2DRcd"
#                                     string tag = "probBTagPDF2D_tag"    },
#                                   {string record = "BTagTrackProbability3DRcd"
#                                     string tag = "probBTagPDF3D_tag"    }
#                                    }
#                                   }

import FWCore.ParameterSet.Config as cms

from RecoBTag.ImpactParameter.impactParameter_cff import *


process = cms.Process("analyzer")
#   source = EmptySource {untracked uint32 firstRun=1 }
#untracked PSet maxEvents = {untracked int32 input = 100}
# Message Logger
process.load("FWCore.MessageLogger.MessageLogger_cfi")

process.load("DQMServices.Components.DQMEnvironment_cfi")
process.load("DQMServices.Core.DQM_cfg")

process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("RecoBTag.Configuration.RecoBTag_cff")

process.load("Validation.RecoB.bTagAnalysis_cfi")
process.load("PhysicsTools.JetMCAlgos.CaloJetsMCFlavour_cfi")  
process.bTagValidation.jetMCSrc = 'AK5byValAlgo'
process.bTagValidation.allHistograms = True 

process.load("TrackingTools/TransientTrack/TransientTrackBuilder_cfi")
process.load('Configuration/StandardSequences/GeometryIdeal_cff')
process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
process.load("Configuration.StandardSequences.MagneticField_38T_cff")

#from CondCore.DBCommon.CondDBCommon_cfi import *
#process.load("RecoBTag.TrackProbability.trackProbabilityFakeCond_cfi")
#process.trackProbabilityFakeCond.connect = "sqlite_fip:btagnew_test_startup.db"
#process.es_prefer_trackProbabilityFakeCond = cms.ESPrefer("PoolDBESSource","trackProbabilityFakeCond")


process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1000) )


process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring( 
     '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0007/F6DE1DAA-ED4E-DE11-8A9B-001D09F25325.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0007/F05D9103-524F-DE11-96EB-001D09F2AD4D.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0007/F01E8CB2-F14E-DE11-B2EF-001D09F2A465.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0007/D2AB5809-F14E-DE11-B36C-001D09F24024.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0007/880B0814-EA4E-DE11-8BB4-001D09F276CF.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0007/1C221D94-F04E-DE11-A9ED-001D09F23944.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0000/F01E9FFE-5F4F-DE11-97D5-003048678FE4.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0000/8844A4D8-734E-DE11-9208-001731AF6859.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0000/720A062B-794E-DE11-9E6C-003048767ED5.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0000/407B954B-764E-DE11-86B9-003048678B44.root',
        '/store/relval/CMSSW_3_1_0_pre9/RelValQCD_Pt_80_120/GEN-SIM-RECO/STARTUP_31X_v1/0000/089C14CE-814E-DE11-91EC-003048678B0E.root'

    
    )
)


process.p = cms.Path(process.myPartons* process.iterativeCone5Flavour *process.impactParameterTagInfos
   *process.jetProbabilityBJetTags*process.jetBProbabilityBJetTags*process.bTagValidation*process.dqmSaver)
process.dqmEnv.subSystemFolder = 'BTAG'
process.dqmSaver.producer = 'DQM'
process.dqmSaver.workflow = '/POG/BTAG/BJET'
process.dqmSaver.convention = 'RelVal'
