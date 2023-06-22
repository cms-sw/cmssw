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
process.load( 'Configuration.Geometry.GeometryExtended2026D88Reco_cff' ) 
process.load( 'Configuration.Geometry.GeometryExtended2026D88_cff' )
process.load( 'Configuration.StandardSequences.MagneticField_cff' )
process.load( 'Configuration.StandardSequences.FrontierConditions_GlobalTag_cff' )
process.load( 'L1Trigger.TrackTrigger.TrackTrigger_cff' )

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase2_realistic', '')

# load code that associates stubs with mctruth
process.load( 'SimTracker.TrackTriggerAssociation.StubAssociator_cff' )
# load code that produces DTCStubs
process.load( 'L1Trigger.TrackerDTC.ProducerED_cff' )
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
#--- Load code that analyzes tfp Stubs
process.load( 'L1Trigger.TrackerTFP.Analyzer_cff' )

# build schedule
process.mc = cms.Sequence( process.StubAssociator )
process.dtc = cms.Sequence( process.TrackerDTCProducer + process.TrackerDTCAnalyzer )
process.gp = cms.Sequence( process.TrackerTFPProducerGP + process.TrackerTFPAnalyzerGP )
process.ht = cms.Sequence( process.TrackerTFPProducerHT + process.TrackerTFPAnalyzerHT )
process.mht = cms.Sequence( process.TrackerTFPProducerMHT + process.TrackerTFPAnalyzerMHT )
process.zht = cms.Sequence( process.TrackerTFPProducerZHT + process.TrackerTFPAnalyzerZHT )
process.interIn = cms.Sequence( process.TrackerTFPProducerZHTout + process.TrackerTFPProducerKFin + process.TrackerTFPAnalyzerKFin )
process.kf = cms.Sequence( process.TrackerTFPProducerKF + process.TrackerTFPAnalyzerKF )
process.interOut = cms.Sequence( process.TrackerTFPProducerTT + process.TrackerTFPProducerAS )#+ process.TrackerTFPAnalyzerTT )
process.tt = cms.Path( process.mc + process.dtc + process.gp + process.ht + process.mht + process.zht + process.interIn + process.kf )#+ process.interOut )
process.schedule = cms.Schedule( process.tt )

# create options
import FWCore.ParameterSet.VarParsing as VarParsing
options = VarParsing.VarParsing( 'analysis' )
# specify input MC
Samples = [
'/store/relval/CMSSW_12_6_0_pre4/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU_125X_mcRun4_realistic_v2_2026D88PU200-v1/2590000/00b3d04b-4c7b-4506-8d82-9538fb21ee19.root'
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
  #skipEvents = cms.untracked.uint32( 3 + 8 ),
  noEventSort = cms.untracked.bool( True ),
  secondaryFileNames = cms.untracked.vstring(),
  duplicateCheckMode = cms.untracked.string( 'noDuplicateCheck' )
)
#process.Timing = cms.Service( "Timing", summaryOnly = cms.untracked.bool( True ) )
process.MessageLogger.cerr.enableStatistics = False
process.TFileService = cms.Service( "TFileService", fileName = cms.string( "Hist.root" ) )
