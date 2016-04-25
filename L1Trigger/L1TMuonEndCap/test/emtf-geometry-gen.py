#!/usr/bin/env python
import FWCore.ParameterSet.Config as cms

process = cms.Process("ANALYSIS")

process.maxEvents = cms.untracked.PSet(
	input = cms.untracked.int32(1)
)

process.source = cms.Source("PoolSource",



###the file used at the moment will probably not generate what I want here###

#fileNames = cms.untracked.vstring('file:/raid/raid5/cms/ikfuric/CMSSW_2_2_5/src/L1Trigger/CSCTrackFinder/test/GenNegMuons-10k-AllEta.root')
#fileNames = cms.untracked.vstring('file:/home/madorsky/CMSSW_2_2_12/src/L1Trigger/CSCTrackFinder/test/GenNegMuons-10k-AllEta.root')
fileNames = cms.untracked.vstring('file:/afs/cern.ch/work/a/abrinke1/public/EMTF/Emulator/samples/ZMu-PromptReco-v4/0EFE474F-D26B-E511-9618-02163E011F4B.root')

# ('file:/home/brianwilliams/CMSSW_2_2_6/src/SLHCUpgradeSimulations/L1Trigger/test/outFile.root')
# ('file:/home/mfisher/dir/GenPosMuonsL.root')
# fileNames = cms.untracked.vstring('file:/home/gartner/CMSSW_2_1_0_pre5/src/L1Trigger/CSCTrackFinder/test/SingleMu2to100.root')#,
#'/home/gartner/CMSSW_2_1_0_pre5/src/L1Trigger/CSCTrackFinder/test/SingleMu50to100.root' )
)

# process.load("FWCore.MessageLogger.MessageLogger_cfi")
# process.load("Geometry.MuonCommonData.muonIdealGeometryXML_cfi")
# process.load("Geometry.CSCGeometry.cscGeometry_cfi")
# process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")


#process.load('Configuration/StandardSequences/FrontierConditions_GlobalTag_cff')
#process.GlobalTag.globaltag="MC_72_V1"
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc', '')
#process.GlobalTag.globaltag="MC_31X_V1::All"
#process.GlobalTag.globaltag="GR09_P_V2::All"
#process.load("Configuration/StandardSequences/Geometry_cff") 
process.load('Configuration.Geometry.GeometryExtended2015Reco_cff')
process.load("Configuration.StandardSequences.MagneticField_cff")

# To get a GlobalPositionRcd
# ##########################

# process.load("CondCore.DBCommon.CondDBSetup_cfi")

#process.PoolDBESSource = cms.ESSource("PoolDBESSource", process.CondDBSetup, connect = cms.string("sqlite_file:inertGlobalPositionRcd.db"), toGet = #cms.VPSet(cms.PSet(record = cms.string("DTAlignmentRcd"), tag = cms.string("DTAlignmentRcd")),
#cms.PSet(record = cms.string("DTAlignmentErrorRcd"), tag = cms.string("DTAlignmentErrorRcd")), cms.PSet(record = cms.string("CSCAlignmentRcd"), tag = #cms.string("CSCAlignmentRcd")), cms.PSet(record = cms.string("CSCAlignmentErrorRcd"), tag = cms.string("CSCAlignmentErrorRcd"))))
# process.PoolDBESSource = cms.ESSource("PoolDBESSource", process.CondDBSetup, connect = cms.string("sqlite_file:inertGlobalPositionRcd.db"), toGet = cms.VPSet(cms.PSet(record = cms.string("GlobalPositionRcd"), tag = cms.string("inertGlobalPositionRcd")), cms.PSet(record = cms.string("CSCAlignmentRcd"), tag = cms.string("CSCAlignmentRcd"))))
#process.PoolDBESSource = cms.ESSource("PoolDBESSource", process.CondDBSetup, connect = cms.string("sqlite_file:inertGlobalPositionRcd.db"), toGet = cms.VPSet(cms.PSet(record = cms.string("CSCAlignmentRcd"), tag = cms.string("CSCAlignmentRcd")), cms.PSet(record = cms.string("CSCAlignmentErrorRcd"), tag = cms.string("CSCAlignmentErrorRcd"))))
#process.CSCGeometryESModule.applyAlignment = False

# Analysis Module Definition
############################
process.effic = cms.EDAnalyzer("slhc_geometry",
	OutFile = cms.untracked.string("Outfile.root"),
	lutParam = cms.PSet()
)
	
# Path Definition
#################
process.p = cms.Path(process.effic)





# old crap
# process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
# process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
# process.load("Geometry.CMSCommonData.cmsIdealGeometryXML_cfi")
# process.load("Geometry.MuonNumbering.muonNumberingInitialization_cfi")
# # include "CondCore/DBCommon/data/CondDBSetup.cfi"
# es_source inertGlobalPositionRcd = PoolDBESSource {
#     using CondDBSetup
#     string connect = "sqlite_file:inertGlobalPositionRcd.db"
#     VPSet toGet = {
#         {
#             string record = "GlobalPositionRcd"
#             string tag = "inertGlobalPositionRcd"
#         }
#     }
# }

# process.load("CondCore.DBCommon.CondDBSetup_cfi")
# 
#  process.PoolDBOutputService = cms.Service("PoolDBOutputService",
#      process.CondDBSetup,
#      # Writing to oracle needs the following shell variable setting (in zsh):
#      # export CORAL_AUTH_PATH=/afs/cern.ch/cms/DB/conddb
#      # string connect = "oracle://cms_orcoff_int2r/CMS_COND_ALIGNMENT"
#      timetype = cms.untracked.string('runnumber'),
#      connect = cms.string('sqlite_file:output.db'),
#      # untracked uint32 authenticationMethod = 1
#      toPut = cms.VPSet(cms.PSet(
#          record = cms.string('GlobalPositionRcd'),
#          tag = cms.string('IdealGeometry')
#      ))
#  )



#  process.GlobalPositionRcdWrite = cms.EDFilter("GlobalPositionRcdWrite",
#      hcal = cms.PSet(
#          beta = cms.double(),
#          alpha = cms.double(),
#          y = cms.double(),
#          x = cms.double(),
#          z = cms.double(),
#          gamma = cms.double()
#      ),
#      muon = cms.PSet(
#          beta = cms.double(),
#          alpha = cms.double(),
#          y = cms.double(),
#          x = cms.double(),
#          z = cms.double(),
#          gamma = cms.double()
#      ),
#      tracker = cms.PSet(
#          beta = cms.double(),
#          alpha = cms.double(),
#          y = cms.double(),
#          x = cms.double(),
#          z = cms.double(),
#          gamma = cms.double()
#      ),
#      ecal = cms.PSet(
#          beta = cms.double(),
#          alpha = cms.double(),
#          y = cms.double(),
#          x = cms.double(),
#          z = cms.double(),
#          gamma = cms.double()
#      )
# 	 )

# process.inertGlobalPositionRcd = cms.ESSource("PoolDBESSource", process.CondDBSetup, connect = cms.string("sqlite_file:inertGlobalPositionRcd.db"),
# toGet = cms.VPSet(cms.PSet(record = cms.string("GlobalPositionRcd"), tag = cms.string("inertGlobalPositionRcd"))))

# process.inertGlobalPositionRcd = cms.ESSource("PoolDBESSource", process.CondDBSetup, connect = cms.string("sqlite_file:inertGlobalPositionRcd.db"),
# 	toGet = cms.VPSet(cms.PSet(record = cms.string("GlobalPositionRcd"), tag = cms.string("inertGlobalPositionRcd"))))
# process.es_prefer_inertGlobalPositionRcd = cms.ESPrefer("PoolDBESSource", "inertGlobalPositionRcd")

# process.p = cms.Path(process.simCscTriggerPrimitiveDigis*process.simDtTriggerPrimitiveDigis*process.simCsctfTrackDigis*process.simCsctfDigis*process.effic)

# 
# # Event Setup
# #############
# process.load("Configuration.StandardSequences.MagneticField_cff")
# process.load("Configuration.StandardSequences.Geometry_cff")
# process.load("CalibMuon.Configuration.DT_FakeConditions_cff")
# 
# # L1 Emulator
# # ###########
# process.load("Configuration.StandardSequences.SimL1Emulator_cff")
# process.load("L1TriggerConfig.DTTPGConfigProducers.L1DTTPGConfig_cff")
# process.load("L1TriggerConfig.DTTrackFinder.L1DTTrackFinderConfig_cff")
# process.load("L1TriggerConfig.L1CSCTPConfigProducers.L1CSCTriggerPrimitivesConfig_cff")
# process.load("L1TriggerConfig.CSCTFConfigProducers.L1CSCTFConfig_cff")
# process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerScalesConfig_cff")
# process.load("L1TriggerConfig.L1ScalesProducers.L1MuTriggerPtScaleConfig_cff")
# 
# # LUT Producer
# # ############
# process.load("L1TriggerConfig.CSCTFConfigProducers.CSCTFConfigProducer_cfi")
