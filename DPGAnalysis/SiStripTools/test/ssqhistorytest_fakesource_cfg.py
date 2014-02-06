import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing


from PhysicsTools.PatAlgos.tools.helpers import cloneProcessingSnippet

process = cms.Process("sistripqualityhistoryfakesource")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "DONOTEXIST::All",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")
options.register ('firstRun',
                  "1",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.int,          # string, int, or float
                  "First Run Number")

options.parseArguments()

process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    fileMode = cms.untracked.string("FULLMERGE")
    )

process.load("FWCore.MessageService.MessageLogger_cfi")

#----------------------------------------------------------------

process.MessageLogger.cout.placeholder = cms.untracked.bool(False)
process.MessageLogger.cout.threshold = cms.untracked.string("INFO")
process.MessageLogger.cout.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
process.MessageLogger.cout.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(10000)
    )

process.MessageLogger.cerr.placeholder = cms.untracked.bool(False)
process.MessageLogger.cerr.threshold = cms.untracked.string("WARNING")
process.MessageLogger.cerr.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
process.MessageLogger.cerr.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(100000)
    )

#----Remove too verbose PrimaryVertexProducer

process.MessageLogger.suppressInfo.append("pixelVerticesAdaptive")
process.MessageLogger.suppressInfo.append("pixelVerticesAdaptiveNoBS")

#----Remove too verbose BeamSpotOnlineProducer

process.MessageLogger.suppressInfo.append("testBeamSpot")
process.MessageLogger.suppressInfo.append("onlineBeamSpot")
process.MessageLogger.suppressWarning.append("testBeamSpot")
process.MessageLogger.suppressWarning.append("onlineBeamSpot")

#----Remove too verbose TrackRefitter

process.MessageLogger.suppressInfo.append("newTracksFromV0")
process.MessageLogger.suppressInfo.append("newTracksFromOtobV0")

process.MessageLogger.suppressWarning.append("consecutiveHEs")


#------------------------------------------------------------------

#process.MessageLogger.cout.threshold = cms.untracked.string("DEBUG")
#process.MessageLogger.debugModules = cms.untracked.vstring("eventtimedistribution")

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(options.maxEvents) )

process.source = cms.Source("EmptySource",
                            firstRun = cms.untracked.uint32(options.firstRun),
                            numberEventsInRun = cms.untracked.uint32(1)#,
#                            processingMode = cms.untracked.string("Runs")
                            )

#process.source = cms.Source("PoolSource",
#                    fileNames = cms.untracked.vstring(options.inputFiles),
##                    skipBadFiles = cms.untracked.bool(True),
#                    inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
#                    )


process.load("DPGAnalysis.SiStripTools.sistripqualityhistory_noDCS_cff")

process.ssqhistory.runProcess = cms.bool(True)
process.ssqhistory.maxLSBeforeRebin = cms.untracked.uint32(100)
process.ssqhistory.startingLSFraction = cms.untracked.uint32(1)
process.ssqhistorystrips = process.ssqhistory.clone(granularityMode = cms.untracked.uint32(3))

#process.load("DPGAnalysis.SiStripTools.fedbadmodulefilter_cfi")
#process.fedbadmodulefilter.badModThr = cms.uint32(0)
#process.fedbadmodulefilter.wantedHisto = cms.untracked.bool(True)

process.load("DPGAnalysis.SiStripTools.sipixelqualityhistory_cfi")

process.spqhistory.runProcess = cms.bool(True)
process.spqhistory.maxLSBeforeRebin = cms.untracked.uint32(100)
process.spqhistory.startingLSFraction = cms.untracked.uint32(1)
process.spqhistorymod = process.spqhistory.clone(granularityMode = cms.untracked.uint32(1))

process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('ssqhistorytest.root')
                                   )

#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
process.GlobalTag.globaltag = options.globalTag

#


process.p0 = cms.Path(process.ssqhistory+process.ssqhistory+process.ssqhistorystrips + process.spqhistory+process.spqhistorymod) # + process.fedbadmodulefilter)
