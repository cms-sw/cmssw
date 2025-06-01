################################################################################################
# Run bit-accurate TMTT L1 tracking emulation. 
#
# To run execute do
# cmsRun L1Trigger/L1TTrackerTFP/test/test_cfg.py
# where the arguments take default values if you don't specify them. You can change defaults below.
#################################################################################################

import FWCore.ParameterSet.Config as cms

process = cms.Process( "Demo" )
process.load( 'FWCore.MessageService.MessageLogger_cfi' )
process.load( 'Configuration.Geometry.GeometryExtendedRun4D110Reco_cff' ) 
process.load( 'Configuration.Geometry.GeometryExtendedRun4D110_cff' )
process.load( 'Configuration.StandardSequences.MagneticField_cff' )
process.load( 'Configuration.StandardSequences.Services_cff' )
process.load( 'Configuration.EventContent.EventContent_cff' )
process.load( 'Configuration.StandardSequences.EndOfProcess_cff' )
process.load( 'Configuration.StandardSequences.FrontierConditions_GlobalTag_cff' )
process.load( 'L1Trigger.TrackTrigger.TrackTrigger_cff' )

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto::phase2_realistic', '')

# load code that associates stubs with mctruth
process.load( 'SimTracker.TrackTriggerAssociation.StubAssociator_cff' )
# load code that produces DTCStubs
process.load( 'L1Trigger.TrackerDTC.DTC_cff' )
# load code that analyzes DTCStubs
process.load( 'L1Trigger.TrackerDTC.Analyzer_cff' )
# cosutmize TT algorithm
from L1Trigger.TrackerDTC.Customize_cff import *
producerUseTMTT( process )
analyzerUseTMTT( process )
#--- Load code that produces tfp Stubs
process.load( 'L1Trigger.TrackerTFP.Producer_cff' )
from L1Trigger.TrackerTFP.Customize_cff import *
setupUseTMTT( process )
simUseTMTT( process )
#--- Load code that analyzes tfp Stubs
process.load( 'L1Trigger.TrackerTFP.Analyzer_cff' )

# build schedule
process.mc  = cms.Sequence( process.StubAssociator )
process.dtc = cms.Sequence( process.ProducerDTC + process.AnalyzerDTC )
process.pp  = cms.Sequence( process.ProducerPP                        )
process.gp  = cms.Sequence( process.ProducerGP  + process.AnalyzerGP  )
process.ht  = cms.Sequence( process.ProducerHT  + process.AnalyzerHT  )
process.ctb = cms.Sequence( process.ProducerCTB + process.AnalyzerCTB )
process.kf  = cms.Sequence( process.ProducerKF  + process.AnalyzerKF  )
process.dr  = cms.Sequence( process.ProducerDR  + process.AnalyzerDR  )
process.tq  = cms.Sequence( process.ProducerTQ                        )
process.tfp = cms.Sequence( process.ProducerTFP                       )
process.tt  = cms.Path( process.mc + process.dtc + process.pp + process.gp + process.ht + process.ctb)# + process.kf )#+ process.dr + process.tq )# + process.tfp )
process.schedule = cms.Schedule( process.tt )

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
  #skipEvents = cms.untracked.uint32( 30 ),
  noEventSort = cms.untracked.bool( True ),
  secondaryFileNames = cms.untracked.vstring(),
  duplicateCheckMode = cms.untracked.string( 'noDuplicateCheck' ),
)
process.Timing = cms.Service( "Timing", summaryOnly = cms.untracked.bool( True ) )
process.MessageLogger.cerr.enableStatistics = False
process.TFileService = cms.Service( "TFileService", fileName = cms.string( "Hist.root" ) )
