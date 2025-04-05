################################################################################################
# runs DTC stub emulation, plots performance & stub occupancy
# To run execute do
# cmsRun L1Trigger/TrackerDTC/test/test.py
# where the arguments take default values if you don't specify them. You can change defaults below.
#################################################################################################

import FWCore.ParameterSet.Config as cms

process = cms.Process( "Demo" )
process.load( 'FWCore.MessageService.MessageLogger_cfi' )
process.load( 'Configuration.Geometry.GeometryExtended2026D98Reco_cff' ) 
process.load( 'Configuration.Geometry.GeometryExtended2026D98_cff' )
process.load( 'Configuration.StandardSequences.MagneticField_cff' )
process.load( 'Configuration.StandardSequences.FrontierConditions_GlobalTag_cff' )
process.load( 'L1Trigger.TrackTrigger.TrackTrigger_cff' )

from Configuration.AlCa.GlobalTag import GlobalTag
# Temporary change until we switch to D110 geometry.
#process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')
process.GlobalTag = GlobalTag(process.GlobalTag, '140X_mcRun4_realistic_v3', '')

# load code that produces DTCStubs
process.load( 'L1Trigger.TrackerDTC.DTC_cff' )
# load code that analyzes DTCStubs
process.load( 'L1Trigger.TrackerDTC.Analyzer_cff' )
# cosutmize TT algorithm
from L1Trigger.TrackerDTC.Customize_cff import *
#producerUseTMTT(process)
#analyzerUseTMTT(process)

# build schedule (not essential to rerun producer)
process.produce = cms.Path( process.ProducerDTC )
process.analyze = cms.Path( process.AnalyzerDTC )
process.schedule = cms.Schedule( process.produce, process.analyze )

# create options
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing( 'analysis' )
# specify input MC
Samples = [ '/store/relval/CMSSW_14_0_0_pre2/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_133X_mcRun4_realistic_v1_STD_2026D98_PU200_RV229-v1/2580000/0b2b0b0b-f312-48a8-9d46-ccbadc69bbfd.root', '/store/relval/CMSSW_14_0_0_pre2/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_133X_mcRun4_realistic_v1_STD_2026D98_PU200_RV229-v1/2580000/0c3cb20d-8556-450d-b4f0-e5c754818f74.root', '/store/relval/CMSSW_14_0_0_pre2/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_133X_mcRun4_realistic_v1_STD_2026D98_PU200_RV229-v1/2580000/0eafa2b4-711a-43ec-be1c-7e564c294a9a.root',
'/store/relval/CMSSW_14_0_0_pre2/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_133X_mcRun4_realistic_v1_STD_2026D98_PU200_RV229-v1/2580000/1450b1bb-171e-495e-a767-68e2796d95c2.root,' '/store/relval/CMSSW_14_0_0_pre2/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_133X_mcRun4_realistic_v1_STD_2026D98_PU200_RV229-v1/2580000/15498564-9cf0-4219-aab7-f97b3484b122.root',
'/store/relval/CMSSW_14_0_0_pre2/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_133X_mcRun4_realistic_v1_STD_2026D98_PU200_RV229-v1/2580000/1838a806-316b-4f53-9d22-5b3856019623.root' 
]
options.register( 'inputMC', Samples, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Files to be processed" )
# specify number of events to process.
options.register( 'Events',100,VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Number of Events to analyze" )
options.parseArguments()

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.Events) )
process.source = cms.Source(
  "PoolSource",
  fileNames = cms.untracked.vstring( options.inputMC ),
  #skipEvents = cms.untracked.uint32( 47 ),
  secondaryFileNames = cms.untracked.vstring(),
  duplicateCheckMode = cms.untracked.string( 'noDuplicateCheck' )
)
process.Timing = cms.Service( "Timing", summaryOnly = cms.untracked.bool( True ) )
process.TFileService = cms.Service( "TFileService", fileName = cms.string( "Hist.root" ) )

# uncomment next 8 lines to use local cabling map
#process.load("CondCore.CondDB.CondDB_cfi")
#process.CondDB.connect = 'sqlite_file:__PATH_TO_DB__/__FILE_NAME__.db'
#process.PoolDBESSource = cms.ESSource(
#  "PoolDBESSource", process.CondDB, toGet = cms.VPSet( cms.PSet(
#    record = cms.string( 'TrackerDetToDTCELinkCablingMapRcd' ),
#    tag    = cms.string( "__CHOSEN_TAG__"                    )
#) ) )
#process.es_prefer_local_TrackerDetToDTCELinkCablingMapRcd = cms.ESPrefer( "PoolDBESSource", "" )
