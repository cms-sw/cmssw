################################################################################################
# To run execute do
# cmsRun L1Trigger/TrackerDTC/test/test.py
# where the arguments take default values if you don't specify them. You can change defaults below.
#################################################################################################

import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process( "Demo" )

process.load( 'Configuration.Geometry.GeometryExtended2026D49Reco_cff' )
process.load( 'Configuration.Geometry.GeometryExtended2026D49_cff' )
process.load( 'Configuration.StandardSequences.MagneticField_cff' )
process.load( 'Configuration.StandardSequences.FrontierConditions_GlobalTag_cff' )
process.load( 'Configuration.StandardSequences.L1TrackTrigger_cff' )

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag( process.GlobalTag, 'auto:phase2_realistic', '' )

# uncomment next 8 lines to use local cabling map
#process.load("CondCore.CondDB.CondDB_cfi")
#process.CondDB.connect = 'sqlite_file:__PATH_TO_DB__/__FILE_NAME__.db'
#process.PoolDBESSource = cms.ESSource(
#  "PoolDBESSource", process.CondDB, toGet = cms.VPSet( cms.PSet(
#    record = cms.string( 'TrackerDetToDTCELinkCablingMapRcd' ),
#    tag    = cms.string( "__CHOSEN_TAG__"                    )
#) ) )
#process.es_prefer_local_TrackerDetToDTCELinkCablingMapRcd = cms.ESPrefer( "PoolDBESSource", "" )

process.load( "FWCore.MessageLogger.MessageLogger_cfi" )

options = VarParsing.VarParsing( 'analysis' )

#--- Specify input MC
options.register( 'inputMC', 
  '/store/relval/CMSSW_11_1_0_pre1/RelValTTbar_14TeV/GEN-SIM-DIGI-RAW/PU25ns_110X_mcRun4_realistic_v2_2026D49PU200_ext1-v1/20000/0330453B-9B8E-CA41-88B0-A047B68D1AF9.root',
  VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.string, "Files to be processed"
)

#--- Specify number of events to process.
options.register( 'Events',100,VarParsing.VarParsing.multiplicity.singleton, VarParsing.VarParsing.varType.int, "Number of Events to analyze" )

#--- Specify whether to output a GEN-SIM-DIGI-RAW dataset containing the DTCStub Collections.
options.register( 'outputDataset' ,0 ,VarParsing.VarParsing.multiplicity.singleton ,VarParsing.VarParsing.varType.int, "Create GEN-SIM-DIGI-RAW dataset containing DTCStub Collections" )

options.parseArguments()

#--- input and output

process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(False) )

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.Events) )

process.source = cms.Source(
  "PoolSource",
  fileNames = cms.untracked.vstring( options.inputMC ),
  secondaryFileNames = cms.untracked.vstring(),
  duplicateCheckMode = cms.untracked.string( 'noDuplicateCheck' )
)

process.Timing = cms.Service( "Timing", summaryOnly = cms.untracked.bool( True ) )

#--- Load code that produces DTCStubs
process.load( 'L1Trigger.TrackerDTC.Producer_cff' )

process.p = cms.Path( process.TrackerDTCProducer )

# Optionally create output GEN-SIM-DIGI-RAW dataset containing DTC TTStub Collections.
if options.outputDataset == 1:

  # Write output dataset
  process.load( 'Configuration.EventContent.EventContent_cff' )

  process.writeDataset = cms.OutputModule(
    "PoolOutputModule",
    splitLevel = cms.untracked.int32( 0 ),
    eventAutoFlushCompressedSize = cms.untracked.int32( 5242880 ),
    outputCommands = process.RAWSIMEventContent.outputCommands,
    fileName = cms.untracked.string( 'output_dataset.root' ), ## ADAPT IT ##
    dataset  = cms.untracked.PSet(
      filterName = cms.untracked.string(''),
      dataTier   = cms.untracked.string( 'GEN-SIM' )
    )
  )
  # Include DTC TTStub Collections
  process.writeDataset.outputCommands.append( 'keep  *_TrackerDTCProducer_StubAccepted_*' )
  process.pd = cms.EndPath( process.writeDataset )

  process.schedule = cms.Schedule( process.p, process.pd )
