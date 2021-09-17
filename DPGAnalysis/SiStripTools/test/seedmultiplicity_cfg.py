import FWCore.ParameterSet.Config as cms
import FWCore.ParameterSet.VarParsing as VarParsing

process = cms.Process("SeedMultiplicity")

#prepare options

options = VarParsing.VarParsing("analysis")

options.register ('globalTag',
                  "DONOTEXIST",
                  VarParsing.VarParsing.multiplicity.singleton, # singleton or list
                  VarParsing.VarParsing.varType.string,          # string, int, or float
                  "GlobalTag")
#options.globalTag = "DONOTEXIST::All"

options.parseArguments()

#
process.options = cms.untracked.PSet(
    wantSummary = cms.untracked.bool(True),
    fileMode = cms.untracked.string("FULLMERGE")
    )

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.cout.enable = cms.untracked.bool(True)
process.MessageLogger.cout.threshold = cms.untracked.string("WARNING")
process.MessageLogger.cout.default = cms.untracked.PSet(
    limit = cms.untracked.int32(10000000)
    )
process.MessageLogger.cout.FwkReport = cms.untracked.PSet(
    reportEvery = cms.untracked.int32(10000)
    )

process.MessageLogger.cerr.enable = cms.untracked.bool(True)
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


#------------------------------------------------------------------

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.source = cms.Source("PoolSource",
                    fileNames = cms.untracked.vstring(options.inputFiles),
#                    skipBadFiles = cms.untracked.bool(True),
                    inputCommands = cms.untracked.vstring("keep *", "drop *_MEtoEDMConverter_*_*")
                    )


process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")

from Configuration.GlobalRuns.reco_TLR_41X import customisePPData
process=customisePPData(process)
    

process.load("DPGAnalysis.SiStripTools.sipixelclustermultiplicityprod_cfi")
process.load("DPGAnalysis.SiStripTools.sistripclustermultiplicityprod_cfi")
process.seqMultProd = cms.Sequence(process.spclustermultprod+process.ssclustermultprod)

process.load("DPGAnalysis.SiStripTools.multiplicitycorr_cfi")
process.multiplicitycorr.correlationConfigurations = cms.VPSet(
    cms.PSet(xMultiplicityMap = cms.InputTag("ssclustermultprod"),
             xDetSelection = cms.uint32(0), xDetLabel = cms.string("TK"), xBins = cms.uint32(1000), xMax=cms.double(50000), 
             yMultiplicityMap = cms.InputTag("spclustermultprod"),
             yDetSelection = cms.uint32(0), yDetLabel = cms.string("Pixel"), yBins = cms.uint32(1000), yMax=cms.double(20000),
             rBins = cms.uint32(200), scaleFactor =cms.untracked.double(5.))
    )


process.load("DPGAnalysis.SiStripTools.seedmultiplicitymonitor_cfi")
process.seedmultiplicitymonitor.multiplicityCorrelations = cms.VPSet(
    cms.PSet(multiplicityMap = cms.InputTag("ssclustermultprod"),
             detSelection = cms.uint32(0), detLabel = cms.string("TK"), nBins = cms.uint32(1000), nBinsEta = cms.uint32(100), maxValue=cms.double(100000) 
             ),
    cms.PSet(multiplicityMap = cms.InputTag("spclustermultprod"),
             detSelection = cms.uint32(0), detLabel = cms.string("Pixel"), nBins = cms.uint32(1000), nBinsEta = cms.uint32(100), maxValue=cms.double(20000) 
             )
    )
process.p0 = cms.Path(process.siPixelRecHits + process.ckftracks + process.seqMultProd + process.multiplicitycorr + process.seedmultiplicitymonitor )

#----GlobalTag ------------------------

process.load("Configuration.StandardSequences.FrontierConditions_GlobalTag_cff")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, options.globalTag, '')


process.TFileService = cms.Service('TFileService',
                                   fileName = cms.string('seedmultiplicity.root')
                                   )

#print process.dumpPython()
