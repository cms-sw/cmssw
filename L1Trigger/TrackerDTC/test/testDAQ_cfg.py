################################################################################################
# plots cluster occupancy
# To run execute do
# cmsRun L1Trigger/TrackerDTC/test/testDAQ.py
# where the arguments take default values if you don't specify them. You can change defaults below.
#################################################################################################

import FWCore.ParameterSet.Config as cms

process = cms.Process( "Demo" )
process.load( 'FWCore.MessageService.MessageLogger_cfi' )
process.load( 'Configuration.Geometry.GeometryExtended2026D110Reco_cff' ) 
process.load( 'Configuration.Geometry.GeometryExtended2026D110_cff' )
process.load( 'Configuration.StandardSequences.MagneticField_cff' )
process.load( 'Configuration.StandardSequences.FrontierConditions_GlobalTag_cff' )
process.load( 'L1Trigger.TrackTrigger.TrackTrigger_cff' )

from Configuration.AlCa.GlobalTag import GlobalTag
# Temporary change until we switch to D110 geometry.
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

# load code that produces DTCStubs
process.load( 'L1Trigger.TrackerDTC.DTC_cff' )
# load code that analyzes TTCluster
process.load( 'L1Trigger.TrackerDTC.AnalyzerDAQ_cff' )

# build schedule (not essential to rerun producer).
process.produce = cms.Path( process.ProducerDTC )
process.analyzeDAQ = cms.Path( process.TrackerDTCAnalyzerDAQ )
process.schedule = cms.Schedule( process.produce, process.analyzeDAQ )

# create options
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing( 'analysis' )
# specify input MC
Samples = [""]
options.register( 'inputMC', Samples, VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Files to be processed" )
# specify number of events to process.
options.register( 'Events',100,VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Number of Events to analyze" )
options.parseArguments()

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.Events) )
process.source = cms.Source(
  "PoolSource",
  fileNames = cms.untracked.vstring( options.inputMC ),
  secondaryFileNames = cms.untracked.vstring(),
  duplicateCheckMode = cms.untracked.string( 'noDuplicateCheck' )
)
process.Timing = cms.Service( "Timing", summaryOnly = cms.untracked.bool( True ) )
process.TFileService = cms.Service( "TFileService", fileName = cms.string( "Hist.root" ) )
