from __future__ import print_function
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

###################################################################
# Setup 'standard' options
###################################################################
options = VarParsing()

options.register('conditionGT',
                 "auto:run2_data",
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.string,
                 "condition global tag for the job (\"auto:run2_data\" is default)")

options.register('conditionOverwrite',
                 "",
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.string,
                 "configuration to overwrite the condition into the GT (\"\" is default)")

options.register('inputCollection',
                 "ALCARECOSiStripCalMinBias",
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.string,
                 "collections to be used for input (\"ALCARECOSiStripCalMinBias\" is default, use 'generalTracks' for prompt reco and 'ctfWithMaterialTracksP5' for cosmic reco)")

options.register('outputFile',
                 "calibTreeTest.root",
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.string,
                 "name for the output root file (\"calibTreeTest.root\" is default)")

options.register('inputFiles',
                 '/store/data/Run2018D/Cosmics/ALCARECO/SiStripCalCosmics-UL18-v1/40000/0346DCE4-0C70-1344-A7EB-D488B627208C.root',
                 VarParsing.multiplicity.list,
                 VarParsing.varType.string,
                 "file to process")

options.register('maxEvents',
                 -1,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.int,
                 "number of events to process (\"-1\" for all)")

options.register('runNumber',
                 -1,
                 VarParsing.multiplicity.singleton,
                 VarParsing.varType.int,
                 "run number to process (\"-1\" for all)")

options.register('cosmicTriggers', '',
                 VarParsing.multiplicity.list,
                 VarParsing.varType.string,
                 'cosmic triggers')

options.parseArguments()
###################################################################
# To use the prompt reco dataset please use 'generalTracks' as inputCollection
# To use the cosmic reco dataset please use 'ctfWithMaterialTracksP5' as inputCollection


process = cms.Process('CALIB')
process.load('Configuration/StandardSequences/MagneticField_cff')
process.load('Configuration.Geometry.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.conditionGT, options.conditionOverwrite)

process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.StandardSequences.Services_cff')

process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.maxEvents))

process.source = cms.Source("PoolSource", fileNames=cms.untracked.vstring(options.inputFiles))
if options.runNumber != -1:
   if 'Cosmics' not in options.inputCollection:
      print("Restricting to the following events :")
      print('%s:1-%s:MAX'%(options.runNumber,options.runNumber))
      process.source.eventsToProcess = cms.untracked.VEventRange('%s:1-%s:MAX'%(options.runNumber,options.runNumber))
   else:
      print("Restricting to the following lumis for Cosmic runs only:")
      print('%s:1-%s:MAX'%(options.runNumber,options.runNumber))
      process.source.lumisToProcess = cms.untracked.VLuminosityBlockRange('%s:1-%s:MAX'%(options.runNumber,options.runNumber))

process.options = cms.untracked.PSet(
        wantSummary = cms.untracked.bool(True)
        )
process.MessageLogger.cerr.FwkReport.reportEvery = 10000

inTracks = cms.InputTag(options.inputCollection)

process.load('CalibTracker.SiStripCommon.prescaleEvent_cfi')
process.load('CalibTracker.Configuration.Filter_Refit_cff')
## use CalibrationTracks (for clusters) and CalibrationTracksRefit (for tracks)
process.CalibrationTracks.src = inTracks
tracksForCalib = cms.InputTag("CalibrationTracksRefit")

process.prescaleEvent.prescale = 1
process.load("CalibTracker.SiStripCommon.SiStripBFieldFilter_cfi")

from HLTrigger.HLTfilters.triggerResultsFilter_cfi import triggerResultsFilter

process.IsolatedMuonFilter = triggerResultsFilter.clone(
        triggerConditions = cms.vstring("HLT_IsoMu20_*"),
        hltResults = cms.InputTag("TriggerResults", "", "HLT"),
        l1tResults = cms.InputTag(""),
        throw = cms.bool(False)
        )
if len(options.cosmicTriggers) > 0:
    print("Cosmic triggers: {0}".format(", ".join(options.cosmicTriggers)))
    process.IsolatedMuonFilter.triggerConditions = cms.vstring(options.cosmicTriggers)
else:
    print("Cosmic triggers: {0} (default)".format(", ".join(process.IsolatedMuonFilter.triggerConditions)))
    print("Argument passed: {0}".format(options.cosmicTriggers))

process.TkCalSeq = cms.Sequence(
        process.prescaleEvent*
        process.IsolatedMuonFilter*
        process.siStripBFieldOnFilter*
        process.CalibrationTracks,
        cms.Task(process.MeasurementTrackerEvent),
        cms.Task(process.offlineBeamSpot),
        cms.Task(process.CalibrationTracksRefit)
        )

process.load("PhysicsTools.NanoAOD.nano_cff")
process.load("PhysicsTools.NanoAOD.NanoAODEDMEventContent_cff")

## as a test: it should be possible to add tracks fully at configuration level (+ declaring the plugin)
from PhysicsTools.NanoAOD.common_cff import *
## this is equivalent to ShallowTrackProducer as configured for the gain calibration
process.tracksTable = cms.EDProducer("SimpleTrackFlatTableProducer",
        src=tracksForCalib,
        cut=cms.string(""),
        name=cms.string("track"),
        doc=cms.string("SiStripCalMinBias ALCARECO tracks"),
        singleton=cms.bool(False),
        extension=cms.bool(False),
        variables=cms.PSet(
            chi2ndof=Var("chi2()/ndof", float),
            pt=Var("pt()", float),
            hitsvalid=Var("numberOfValidHits()", int), ## unsigned?
            phi=Var("phi()", float),
            eta=Var("eta()", float),
            )
        )
process.load("CalibTracker.SiStripCommon.siStripPositionCorrectionsTable_cfi")
process.siStripPositionCorrectionsTable.Tracks = tracksForCalib
process.load("CalibTracker.SiStripCommon.siStripLorentzAngleRunInfoTable_cfi")

siStripCalCosmicsNanoTables = cms.Task(
        process.nanoMetadata,
        process.tracksTable,
        process.siStripPositionCorrectionsTable,
        process.siStripLorentzAngleRunInfoTable
        )

process.nanoCTPath = cms.Path(process.TkCalSeq, siStripCalCosmicsNanoTables)

process.out = cms.OutputModule("NanoAODOutputModule",
        fileName=cms.untracked.string(options.outputFile),
        outputCommands=process.NANOAODEventContent.outputCommands+[
            "drop edmTriggerResults_*_*_*"
            ],
        SelectEvents=cms.untracked.PSet(
            SelectEvents=cms.vstring("nanoCTPath")
            )
        )
process.end = cms.EndPath(process.out)

process.schedule = cms.Schedule(process.nanoCTPath, process.end)
