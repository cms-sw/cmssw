import FWCore.ParameterSet.Config as cms

process = cms.Process("tester")
process.load("FWCore.MessageLogger.MessageLogger_cfi")
process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string('DEBUG')
process.MessageLogger.debugModules = cms.untracked.vstring('*')

process.source = cms.Source("EmptySource", firstRun = cms.untracked.uint32(276403)) #91
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(1) )

#process.load("L1Trigger.L1TMuon.fakeGmtParams_cff")
process.load("L1Trigger.L1TCalorimeter.caloStage2Params_2016_v2_2_cfi")

# Constructing a Global Tag
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_condDBv2_cff')
from Configuration.AlCa.GlobalTag_condDBv2 import GlobalTag
process.GlobalTag.toGet = cms.VPSet(
 cms.PSet(
           record  = cms.string("L1TUtmTriggerMenuRcd"),
           tag     = cms.string("L1TUtmTriggerMenu_Stage2v0_hlt"),
           connect = cms.string("frontier://FrontierProd/CMS_CONDITIONS")
          ),
 cms.PSet(
           record  = cms.string("L1TMuonBarrelParamsRcd"),
           tag     = cms.string("L1TMuonBarrelParams_Stage2v0_hlt"),
           connect = cms.string("frontier://FrontierPrep/CMS_CONDITIONS")
          ),
 cms.PSet(
           record  = cms.string("L1TMuonOverlapParamsRcd"),
           tag     = cms.string("L1TMuonOverlapParams_Stage2v1_hlt"),
           connect = cms.string("frontier://FrontierPrep/CMS_CONDITIONS")
          ),
 cms.PSet(
           record  = cms.string("L1TMuonEndCapParamsRcd"),
           tag     = cms.string("L1TMuonEndCapParams_Stage2v1_hlt"),
           connect = cms.string("frontier://FrontierPrep/CMS_CONDITIONS")
          ),
 cms.PSet(
           record  = cms.string("L1TMuonGlobalParamsRcd"),
           tag     = cms.string("L1TMuonGlobalParams_Stage2v0_hlt"),
           connect = cms.string("frontier://FrontierPrep/CMS_CONDITIONS")
          ),
 cms.PSet(
           record  = cms.string("L1TCaloParamsRcd"),
           tag     = cms.string("L1TCaloParams_Stage2v0_hlt"),
           connect = cms.string("frontier://FrontierPrep/CMS_CONDITIONS")
          ),
 cms.PSet(
           record  = cms.string("L1TGlobalPrescalesVetosRcd"),
           tag     = cms.string("L1TGlobalPrescalesVetos_Stage2v0_hlt"),
           connect = cms.string("frontier://FrontierPrep/CMS_CONDITIONS")
 )
)

## One can also replace the GT block above with the one below
#from CondCore.DBCommon.CondDBSetup_cfi import CondDBSetup
#process.l1conddb = cms.ESSource("PoolDBESSource",
#       CondDBSetup,
#       connect = cms.string("frontier://FrontierPrep/CMS_CONDITIONS"),
#       toGet   = cms.VPSet(
#            cms.PSet(
#               record  = cms.string("L1TUtmTriggerMenuRcd"),
#               tag     = cms.string("L1TUtmTriggerMenu_Stage2v0_hlt"),
#            ),
#            cms.PSet(
#               record  = cms.string("L1TMuonBarrelParamsRcd"),
#               tag     = cms.string("L1TMuonBarrelParams_Stage2v0_hlt"),
#            ),
#            cms.PSet(
#               record  = cms.string("L1TMuonOverlapParamsRcd"),
#               tag     = cms.string("L1TMuonOverlapParams_Stage2v0_hlt"),
#            ),
#            cms.PSet(
#                 record = cms.string("L1TMuonEndCapParamsRcd"),
#                 tag    = cms.string("L1TMuonEndCapParams_Stage2v1_hlt")
#            ),
#            cms.PSet(
#                 record  = cms.string("L1TMuonGlobalParamsRcd"),
#                 tag     = cms.string("L1TMuonGlobalParams_Stage2v0_hlt"),
#            ),
#            cms.PSet(
#                 record  = cms.string("L1TCaloParamsRcd"),
#                 tag     = cms.string("L1TCaloParams_Stage2v0_hlt"),
#            ),
#            cms.PSet(
#                 record  = cms.string("L1TGlobalPrescalesVetosRcd"),
#                 tag     = cms.string("L1TGlobalPrescalesVetos_Stage2v0_hlt"),
#            )
#       )
#)

# Examples of various home-breed consumers
process.l1cr  = cms.EDAnalyzer("L1MenuReader" )
process.l1or  = cms.EDAnalyzer("L1TOverlapReader", printLayerMap = cms.untracked.bool(True) )
process.l1ecv = cms.EDAnalyzer("L1TEndcapViewer")
process.l1gmr = cms.EDAnalyzer("L1TMuonGlobalParamsViewer", printLayerMap = cms.untracked.bool(True) )
process.l1cpv = cms.EDAnalyzer("L1TCaloParamsViewer", printEgIsoLUT = cms.untracked.bool(False) )
process.l1gpv = cms.EDAnalyzer("L1TGlobalPrescalesVetosViewer",
    prescale_table_verbosity = cms.untracked.int32(1),
    bxmask_map_verbosity     = cms.untracked.int32(1)
)

process.p = cms.Path(process.l1cr + process.l1or + process.l1gmr + process.l1cpv + process.l1gpv)
#process.p = cms.Path(process.l1ecr) <- doesn't work in Prep DB because of the format change https://github.com/cms-sw/cmssw/commit/24e4598354bf66b18bd37a37eb779f35ef563847

