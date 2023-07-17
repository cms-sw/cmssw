# this compares event by event the output of the C++ emulation with the ModelSim simulation of the firmware
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

# load code that produces DTCStubs
process.load( 'L1Trigger.TrackerDTC.ProducerED_cff' )
# cosutmize TT algorithm
#from L1Trigger.TrackerDTC.Customize_cff import *
#producerUseTMTT(process)
#analyzerUseTMTT(process)
#--- Load code that produces tfp Stubs
process.load( 'L1Trigger.TrackerTFP.Producer_cff' )
#--- Load code that demonstrates tfp Stubs
process.load( 'L1Trigger.TrackerTFP.Demonstrator_cff' )

# build schedule
process.tt = cms.Sequence (  process.TrackerDTCProducer
                           + process.TrackerTFPProducerGP
                           + process.TrackerTFPProducerHT
                           + process.TrackerTFPProducerMHT
                           + process.TrackerTFPProducerZHT
                           + process.TrackerTFPProducerZHTout
                           + process.TrackerTFPProducerKFin
                           + process.TrackerTFPProducerKF
                          )
process.demo = cms.Path( process.tt + process.TrackerTFPDemonstrator )
process.schedule = cms.Schedule( process.demo )

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
  #skipEvents = cms.untracked.uint32( 914 ),
  secondaryFileNames = cms.untracked.vstring(),
  duplicateCheckMode = cms.untracked.string( 'noDuplicateCheck' )
)
process.Timing = cms.Service( "Timing", summaryOnly = cms.untracked.bool( True ) )
