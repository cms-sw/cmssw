import FWCore.ParameterSet.Config as cms

process = cms.Process('FibreAnalyser')

process.maxEvents = cms.untracked.PSet ( input = cms.untracked.int32 ( 3564 ) )

#Input file
process.source = cms.Source ( "EmptySource" )

#Logger
#process.MessageLogger = cms.Service (
#    destinations = cms.untracked.vstring ( "detailedInfo.txt" ),
#    detailedInfo.txt = cms.untracked.PSet ( threshold = cms.untracked.string("INFO") ),
#    #debugModules = cms.untracked.vstring ( "l1GctHwDigis", "FibreAnalysis" ),
#    debugModules = cms.untracked.vstring ( "*" ),
#    #suppressWarning = cms.untracked.vstring ( "source", "l1GctHwDigis" )

process.gctRaw = cms.OutputModule( "TextToRaw",
 #Only select one of these at a time
 #filename = cms.untracked.string ( "counter_2008_03_04.dat" )
 #filename = cms.untracked.string ( "logical_id_2008_03_04.dat" )
 #filename = cms.untracked.string ( "jet_counter_2008_05_14.dat" )
 #filename = cms.untracked.string ( "logicalid.dat" )
)

process.l1GctHwDigis = cms.OutputModule( "GctRawToDigi",
  inputLabel = cms.InputTag("gctRaw"),
  gctFedId = cms.int32(745),
  verbose = cms.untracked.bool(False),
  hltMode = cms.bool(False),
  grenCompatibilityMode = cms.bool(False),
  unpackEm = cms.untracked.bool(True),
  unpackJets = cms.untracked.bool(True),
  unpackEtSums = cms.untracked.bool(True),
  unpackInternEm = cms.untracked.bool(True),
  unpackRct = cms.untracked.bool(True),
  unpackFibres = cms.untracked.bool(True)
)

#Fibre Analyzer
process.FibreAnalysis = cms.OutputModule( "GctFibreAnalyzer",
  FibreSource = cms.untracked.InputTag("l1GctHwDigis"),
  #Make sure only one of these are set to True
  doLogicalID = cms.untracked.bool(True),
  doCounter = cms.untracked.bool(False)
)

process.p = cms.Path ( process.gctRaw * process.l1GctHwDigis * process.FibreAnalysis )
